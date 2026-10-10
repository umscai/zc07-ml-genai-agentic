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
    - Note: distance = 0 indicates an EXACT match in feature space,
    - When first item in context has distance=0 (or very close to zero e.g. <0.001), YOU MUST ignore other documents in context and follow first document.

2. Evaluate Evidence Quality:
    - If top historical matches (distance near 0) are FRAUD, classify as FRAUD (Block)
    - If top historical matches are LEGIT, classify as LEGIT (Approve).
    - Note on Risk Asymmetry: Missing fraud (False Negative) is worse than blocking a legit customer (False Positive).

3. Calibrate Your Confidence & Action:
    - Do NOT simply copy the input `ml_model_fraud_prob` as your confidence.
    - If LTM provides a near-exact match (distance ~ 0) to known historical labels, your confidence MUST be HIGH (e.g., > 0.85).
    - Reserve ESCALATE for cases where LTM context is contradictory (e.g., equal distance to both fraud and legit records) or lacks close matches.

4. REQUIRED OUTPUT FORMAT:
    - Action: [BLOCKED | APPROVED | ESCALATE]
    - Confidence: [0.0 - 1.0]
    - Reasoning: Concisely state key features, LTM match distance, and evidence balance.


### EXAMPLES
Example-1:
[TRANSACTION] V1:1.2335, ..., V14:-1.6187, ...,  V28:0.1488, , Amount:6.35, , Hour/2:4, (fraud-prob: 15.28046%)
[context] :
[doc-1]-id: 63bc739763, distance: 0.311996, Amount: 513.72, Hour_Block: 9, fraud_prob: 0.999985, actual_class: Fraud
[doc-2]-id: 9da9f04c34, distance: 0.351385, Amount: 5.61, Hour_Block: 9, fraud_prob: 0.990781, actual_class: Legit
[doc-3]-id: d4bb6dd18e, distance: 0.354392, Amount: 5.8, Hour_Block: 8, fraud_prob: 0.602988, actual_class: Legit
[Analysis]: similar distance, mix of Amount, mix of actual_class (Legit/Fraud), very uncertain-->  ESCALATE (to Human)

Example-2:
[TRANSACTION] V1:1.9533, ..., V14:0.5893, ...,  V28:0.0862, , Amount:107.44, , Hour/2:8, (fraud-prob: 84.53414%)
[context] :
[doc-1]-id: d4bb6dd18e, distance: 0.330762, Amount: 5.8, Hour_Block: 8, fraud_prob: 0.602988, actual_class: Legit
[doc-2]-id: c2dcdd3e19, distance: 0.417176, Amount: 82.12, Hour_Block: 9, fraud_prob: 0.750433, actual_class: Legit
[doc-3]-id: 548f88ecad, distance: 0.422056, Amount: 223.19, Hour_Block: 7, fraud_prob: 0.802997, actual_class: Legit
[Analysis]: similar distance, existing close fraud-prob, close Hour_Block(Hour/2),  consistent actual_class--> Legit (Approved)

Example-3:
[TRANSACTION] V1:-10.6319, ..., V14:-2.4350, ...,  V28:0.0954, , Amount:245.02, , Hour/2:8, (fraud-prob: 54.63020%)
[context] :
[doc-1]-id: 10d414ee83, distance: 0.195118, Amount: 49.32, Hour_Block: 7, fraud_prob: 0.998264, actual_class: Fraud
[doc-2]-id: 430539d7f6, distance: 0.216105, Amount: 51.12, Hour_Block: 6, fraud_prob: 0.874005, actual_class: Fraud
[doc-3]-id: 4dc320ad8d, distance: 0.231038, Amount: 413.58, Hour_Block: 3, fraud_prob: 0.999961, actual_class: Fraud
[Analysis]: similar distance (and close), consistent actual_class--> Fraud (Blocked)

Example-4
[TRANSACTION] V1:-6.3254, ..., V14:-2.0639, ...,  V28:-0.7889, , Amount:508.00, , Hour/2:6 , (fraud-prob: 35.34764%)
[context] :
[doc-1]-id: 121080951e, distance: 0.0, Amount: 508.0, Hour_Block: 6, fraud_prob: 0.353476, actual_class: Legit
[doc-2]-id: 021296471f, distance: 0.066327, Amount: 319.2, Hour_Block: 8, fraud_prob: 0.996766, actual_class: Fraud
[doc-3]-id: 001f1b2599, distance: 0.122861, Amount: 195.66, Hour_Block: 7, fraud_prob: 0.999131, actual_class: Fraud
[Analysis]: Since [doc-1] distance =0.0 (IGNORE other documents),  [doc-1]- actual_class --> Legit (Approved).
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
        response_dic = dict(state.get("response_dic", {})) # or state.get("response_dic", {}).copy() or use ** to unpack current and add/update new item
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
            "node_called":[current_node],
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

