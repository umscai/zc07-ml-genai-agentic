

from langchain_core.runnables import RunnableConfig
from agents.agent_common import AgentState, GraphNode, GraphAction
from tools.ltm_db_tools import add_single_transaction_to_vector_db
from tools.transaction_db_tools import log_transaction
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def db_update_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    LangGraph Node responsible for logging anomaly information into Long-Term-Memory (LTM)
    Args:
        state (AgentState): Current state of the workflow graph.
        config (RunnableConfig): Runtime configuration (e.g., configurable LLM instance).
    Returns:
        AgentState: Partial state update dict for LangGraph.
    """
    try:
        current_node = GraphNode.dB
        next_node = GraphNode.end
        features = state.get("features", {})
        actual_class = state.get("actual_class", 1)
        pred_class = state.get("pred_class", 1)
        decision_node = state.get("decision_node", GraphNode.human)
        fraud_prob = state.get("fraud_prob", 1.0)
        source_id = state.get("source_id", 'unknown')
        response_dic = dict(state.get("response_dic", {})) # or state.get("response_dic", {}).copy() or use ** to unpack current and add/update new item

        lod_id, timestamp_str =log_transaction(
            transaction = features,
            actual_class = actual_class,
            pred_class = pred_class,
            source_id = source_id
        )
        func_used=['log_transaction']
        response=f'{timestamp_str}:transaction[{lod_id}] logged to dB.'

        if (pred_class == 0) and (actual_class == 0):
            label = 'TRUE_NEGATIVE'
        elif (pred_class == 0) and (actual_class == 1):
            label = 'FALSE_NEGATIVE'
        elif (pred_class == 1) and (actual_class == 0):
            label = 'FALSE_POSITIVE'
        elif (pred_class == 1) and (actual_class == 1):
            label = 'TRUE_POSITIVE'
        else:
            # e.g. actual_label = 2 this scenario will be post process from log_transaction to vector-dB
            label = 'UNKNOWN'

        # We do not log TRUE_NEGATIVE in LTM (vector-dB)
        is_new_insert = None
        if label != 'TRUE_NEGATIVE' and label !='UNKNOWN':
            df1 = pd.DataFrame([features])
            is_new_insert, doc_id = add_single_transaction_to_vector_db(
                df = df1,
                fraud_prob = fraud_prob,
                actual_class = actual_class,
                actual_label= label,
                node = decision_node
            )
            func_used.append('add_single_transaction_to_vector_db')
            if is_new_insert:
                response +=f'{label} with doc-id={doc_id} ADDED to Vector-dB (LTM).'
            else:
                response +=f'{label} with doc-id={doc_id} Updated in Vector-dB (LTM).'
        else:
            response +=f'{label} NOT added to Vector-dB (LTM).'

        response_dic[current_node]=response
        updates = {
            #'actual_class': actual_class,
            'current_node': current_node,
            #'fraud_prob': fraud_prob,
            #'pred_class': pred_class,
            'next_node': next_node,
            #'action': action,
            'response_dic': response_dic,
            "func_used": func_used,
            "node_called":[current_node],
            "timestamp_str": timestamp_str,
            "error": None
        }

        logging_dict = {
            "source_id": source_id,
            "current_node": current_node,
            "is_new_insert": is_new_insert,
            "lod_id": lod_id,
        }
        logger.info(logging_dict)
        return updates
    except Exception as e:
        logger.error(f'Error in db_update_node: {e}')
        updates = {
            current_node: GraphNode.dB,
            "error":  str(e)
        }
        return updates
