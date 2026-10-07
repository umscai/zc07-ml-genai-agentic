import logging
from typing import Literal, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage
from agents.agent_common import AgentState, GraphAction, GraphNode, invoke_react_agent
from utils import get_val
from config import settings

logger = logging.getLogger(__name__)

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
        current_node = GraphNode.llm
        response_dic = state.get("response_dic", {})
        fraud_prob = state.get("fraud_prob", 1.0)
        if fraud_prob < 0.25:
            pred_class = 0
            next_node = GraphNode.approved
            decision_node = GraphNode.llm
            action = GraphAction.approved
            response=f'Legit-Transaction send to {next_node} (fraud-prob: {fraud_prob:.5%})'
        elif fraud_prob > 0.75:
            pred_class = 1
            next_node = GraphNode.blocked
            decision_node = GraphNode.llm
            action = GraphAction.blocked
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

