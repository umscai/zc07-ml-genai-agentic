from langchain_core.runnables import RunnableConfig
from agents.agent_common import AgentState, GraphNode, GraphAction
import logging
logger = logging.getLogger(__name__)


def blocked_procedure_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    LangGraph Node responsible for Block procedure when transaction flagged as Fraud
    [NOTE]: It is a placeholder for expansion in real production
    Args:
        state (AgentState): Current state of the workflow graph.
        config (RunnableConfig): Runtime configuration (e.g., configurable LLM instance).
    Returns:
        AgentState: Partial state update dict for LangGraph.
    """
    try:
        fraud_prob = state.get("fraud_prob", 1.0)
        current_node = GraphNode.blocked
        next_node = GraphNode.dB
        action = GraphAction.blocked
        response_dic = state.get("response_dic", {})
        response=f'Fraud-Transaction {action} (fraud-prob: {fraud_prob:.5%})'
        response_dic[current_node]=response
        updates = {
            #'actual_class': actual_class,
            'current_node': current_node,
            #'fraud_prob': fraud_prob,
            #'pred_class': pred_class,
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
        logger.error(f'Error in blocked_procedure_node: {e}')
        updates = {
            current_node: GraphNode.blocked,
            "error":  str(e)
        }
        return updates
