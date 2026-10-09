from langchain_core.runnables import RunnableConfig
from agents.agent_common import AgentState, GraphNode, GraphAction
import logging
logger = logging.getLogger(__name__)


def human_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    LangGraph Node simulating a human investigator
    It has access to actual_class to finalize the Legit/Fraud classification
    Args:
        state (AgentState): Current state of the workflow graph.
        config (RunnableConfig): Runtime configuration (e.g., configurable LLM instance).
    Returns:
        AgentState: Partial state update dict for LangGraph.
    """
    try:
        current_node = GraphNode.human
        response_dic = state.get("response_dic", {})
        # ORACLE
        actual_class = state.get('actual_class','1')
        fraud_prob = state.get("fraud_prob", 1.0)
        if actual_class == 0:
            pred_class = 0
            next_node = GraphNode.approved
            decision_node = GraphNode.human
            action = GraphAction.approved
            response=f'Legit-Transaction send to {next_node} (fraud-prob: {fraud_prob:.5%})'
        else:
            pred_class = 1
            next_node = GraphNode.blocked
            decision_node = GraphNode.human
            action = GraphAction.blocked
            response=f'Fraud-Transaction send to {next_node} (fraud-prob: {fraud_prob:.5%})'

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
            "node_called":[current_node],
            "error": None
        }
        logger.info(f"response={response}")
        return updates

    except Exception as e:
        logger.error(f'Error in human_node: {e}')
        updates = {
            current_node: GraphNode.human,
            "error":  str(e)
        }
        return updates
