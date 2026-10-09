import logging
from typing import Literal, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage
from agents.agent_common import AgentState, GraphAction, GraphNode, invoke_react_agent
from utils import get_val
from config import settings
from tools.ltm_db_tools import get_top_n_context, get_top_n_context_for_human

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Tools & Structured Output Schema
# ------------------------------------------------------------------------------
TOOLS = [get_top_n_context]


class ClassificationOutput(BaseModel):
    action: Literal["APPROVED", "BLOCKED", "ESCALATE"] = Field(
        description='APPROVED: when transaction is Legit, BLOCKED: when transaction is Fraud , ESCALATE when uncertain.'
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0 on transaction classification"
    )
    context:str = Field(description="context that retrieved from Long-Term Memory (LTM) during investigation")
    reasoning:str = Field(description='Brief 2-3 sentence explanation comparing current features against context.')

# ------------------------------------------------------------------------------
# Prompt Design
# ------------------------------------------------------------------------------
CLASSIFICATION_PROMPT ="""
You are an expert Senior Fraud Risk Analyst operating in an automated triage system.
Your job is to evaluate escalated transactions where the machine learning classifier model was uncertain.

### INPUT DATA
transaction_data: {transaction_data}
ml_model_fraud_prob: {fraud_prob: .4f}

### INSTRUCTION
1. Use `get_top_n_context` tool with transaction_data to retrieve similar historical transactions.
    - Note: distance = 0 indicates an EXACT match in feature space.

2. Evaluate Evidence Quality:
    - If top historical matches (distance near 0) are overwhelmingly FRAUD, classify as FRAUD.
    - If top historical matches are overwhelmingly LEGIT, classify as LEGIT.
    - Note on Risk Asymmetry: Missing fraud (False Negative) is significantly worse than blocking a legit customer (False Positive).

3. Calibrate Your Confidence & Action:
    - Do NOT simply copy the input `ml_model_fraud_prob` as your confidence.
    - If LTM provides a near-exact match (distance ~ 0) to known historical labels, your confidence MUST be HIGH (e.g., > 0.85).
    - Reserve ESCALATE for cases where LTM context is contradictory (e.g., equal distance to both fraud and legit records) or lacks close matches.

4. REQUIRED OUTPUT FORMAT:
    - Action: [BLOCKED | APPROVED | ESCALATE]
    - Confidence: [0.0 - 1.0]
    - Reasoning: Concisely state key features, LTM match distance, and evidence balance.
"""

CLASSIFICATION_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["transaction_data", "fraud_prob"],
    template=CLASSIFICATION_PROMPT
)

# ------------------------------------------------------------------------------
# LangGraph Node Function
# ------------------------------------------------------------------------------
def llm_rag_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    LangGraph Node capable of using react-agent (LLM) and context (RAG)
    from LTM (Chroma-Vector-dB) to analyze transaction for fraud when initial
    ML-model was uncertain
    Args:
        state (AgentState): Current state of the workflow graph.
        config (RunnableConfig): Runtime configuration (e.g., configurable LLM instance).
    Returns:
        AgentState: Partial state update dict for LangGraph.
    """
    try:
        llm = config.get("configurable").get("llm")
        current_node = GraphNode.llm
        response_dic = state.get("response_dic", {})
        fraud_prob = state.get("fraud_prob", 1.0)
        features = state.get("features", {})
        context_for_human, _ = get_top_n_context_for_human(features)

        # Format System Prompt
        sys_message = CLASSIFICATION_PROMPT_TEMPLATE.format(
            transaction_data=features,
            fraud_prob =  fraud_prob
        )
        messages = [SystemMessage(content=sys_message)]

        result, tools_used = invoke_react_agent(ClassificationOutput, messages, llm, TOOLS)

        data = result.get("structured_response") if isinstance(result, dict) else result
        ai_msg = result.get("messages", []) if isinstance(result, dict) else []

        action = get_val(data, "action") or GraphAction.escalate
        llm_reasoning = get_val(data, "reasoning") or ""
        llm_confidence = get_val(data, "confidence") or 0.0
        llm_context = get_val(data, "context") or ""

        logging_dict = {
            "action": action,
            "tools_used": tools_used,
            "llm_confidence": llm_confidence,
        }
        logger.info(logging_dict)


        if action ==  GraphAction.approved:
            pred_class = 0
            next_node = GraphNode.approved
            decision_node = GraphNode.llm
            #action = GraphAction.approved
            response=f'Legit-Transaction send to {next_node} (fraud-prob: {fraud_prob:.5%})'
        elif action ==  GraphAction.blocked:
            pred_class = 1
            next_node = GraphNode.blocked
            decision_node = GraphNode.llm
            #action = GraphAction.blocked
            response=f'Fraud-Transaction send to {next_node} (fraud-prob: {fraud_prob:.5%})'
        else:
            pred_class = 2
            next_node = GraphNode.human
            decision_node = None
            action = GraphAction.escalate
            response=f'{action} to {next_node} for extra investigation (fraud-prob: {fraud_prob:.5%})'

        response_dic[current_node]=response
        updates = {
            #'actual_class': actual_class,
            'current_node': current_node,
            'decision_node': decision_node,
            #'fraud_prob': fraud_prob,
            'pred_class': pred_class,
            'next_node': next_node,
            'action': action,
            'response_dic': response_dic,
            "tools_used": tools_used,
            "func_used": ['get_top_n_context_for_human'],
            "messages": ai_msg,
            "llm_confidence": llm_confidence,
            "llm_reasoning" : llm_reasoning,
            "llm_context": llm_context,
            'context': context_for_human,
            #"func_used": ['get_fraud_prob_from_ml_classifier'],

            "error": None
        }
        logger.info(f"response={response}")
        return updates

    except Exception as e:
        logger.error(f'Error in llm_rag_node: {e}')
        updates = {
            current_node: GraphNode.llm,
            "error":  str(e)
        }
        return updates

