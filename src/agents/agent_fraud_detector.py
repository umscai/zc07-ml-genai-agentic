
from langchain_core.runnables import RunnableConfig
from config import settings
from agents.agent_common import AgentState, GraphAction, GraphNode
from tools.fraud_detector_tool import get_fraud_prob_from_ml_classifier
import logging
import joblib
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------
def fraud_detector_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    This LangGraph Node responsible for processing incoming transaction to check
    if it is a Legit or Fraud transaction using ML classifier model based on two optimized thresholds
    if uncertain , it will be escalated to LLM-Agent with access to Vector-dB for history check
    Args:
        state(AgentState): current state of the LangGraph/Workflow graph
        config(RunnableConfig): Runtime configuration for the runnable (e.g. thread_id)
    Returns:
        AgentState: The updated graph state
    """
    try:
        source_id = state.get("source_id", 'gen')
        features = state.get("features", {})
        response_dic = state.get("response_dic", {})
        actual_class= state.get("actual_class", 2)
        current_node = GraphNode.detector
        #features should not be empty, it will check in assistance before calling the graph
        fraud_prob = get_fraud_prob_from_ml_classifier(features)
        if fraud_prob < settings.LOW_THR:
            pred_class = 0
            next_node = GraphNode.approved
            decision_node = GraphNode.detector
            action = GraphAction.approved
            response=f'Legit-Transaction send to {next_node} (fraud-prob: {fraud_prob:.5%})'
        elif fraud_prob > settings.HIGH_THR:
            pred_class = 1
            next_node = GraphNode.blocked
            decision_node = GraphNode.detector
            action = GraphAction.blocked
            response=f'Fraud-Transaction send to {next_node} (fraud-prob: {fraud_prob:.5%})'
        else:
            pred_class = 2
            next_node = GraphNode.llm
            action = GraphAction.escalate
            decision_node = None
            response=f'{action} to {next_node} for extra investigation (fraud-prob: {fraud_prob:.5%})'

        response_dic[current_node]=response
        updates = {
            'actual_class': actual_class,
            'current_node': current_node,
            'decision_node': decision_node,
            'fraud_prob': fraud_prob,
            'pred_class': pred_class,
            'next_node': next_node,
            'action': action,
            'response_dic': response_dic,
            "func_used": ['get_fraud_prob_from_ml_classifier'],
            "error": None
        }
        logger.info(f"response={response}")
        return updates
    except Exception as e:
        logger.error(f"Error in fraud_detector_node: {str(e)}")
        updates = {
            current_node: GraphNode.detector,
            "error":  str(e)
        }
        return updates