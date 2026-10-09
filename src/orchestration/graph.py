import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agents.agent_common import AgentState, GraphNode, GraphAction
from agents.agent_approved_procedure import approved_procedure_node
from agents.agent_blocked_procedure import blocked_procedure_node
from agents.agent_fraud_detector import fraud_detector_node
from agents.agent_human import human_node
from agents.agent_llm_rag import llm_rag_node
from agents.agent_db_update import db_update_node
from config import settings
import logging
logger = logging.getLogger(__name__)



def route_after_detector(state: AgentState) -> str:
    action = state.get("action", GraphAction.escalate)
    return action # 'APPROVED', 'BLOCKED', 'Escalate'

def route_after_llm(state: AgentState) -> str:
    action = state.get("action", GraphAction.escalate)
    return action # 'APPROVED', 'BLOCKED', 'Escalate'

def route_after_human(state: AgentState) -> str:
    action = state.get("action", GraphAction.blocked)
    return action # 'APPROVED', 'BLOCKED'


def build_fraud_detection_workflow() -> CompiledStateGraph:
    builder = StateGraph(AgentState)

    # Add Nodes
    builder.add_node("agent_ML_Fraud_Detector", fraud_detector_node)
    builder.add_node("agent_LLM_RAG", llm_rag_node)
    builder.add_node("agent_Human", human_node)
    builder.add_node("agent_Approved_Procedure", approved_procedure_node)
    builder.add_node("agent_Blocked_Procedure", blocked_procedure_node)
    builder.add_node("agent_dB_Update", db_update_node)

    # Define Graph Edges
    builder.add_edge(START, "agent_ML_Fraud_Detector")
    builder.add_conditional_edges(
        "agent_ML_Fraud_Detector",
        route_after_detector,
        {
            GraphAction.approved: "agent_Approved_Procedure",
            GraphAction.blocked: "agent_Blocked_Procedure",
            GraphAction.escalate: "agent_LLM_RAG",
        }
    )

    builder.add_conditional_edges(
        "agent_LLM_RAG",
        route_after_llm,
        {
            GraphAction.approved: "agent_Approved_Procedure",
            GraphAction.blocked: "agent_Blocked_Procedure",
            GraphAction.escalate: "agent_Human",
        }
    )

    builder.add_conditional_edges(
            "agent_Human",
            route_after_human,
            {
                GraphAction.approved: "agent_Approved_Procedure",
                GraphAction.blocked: "agent_Blocked_Procedure",
            }
        )

    # 3. Action Nodes Converge Back to dB/LTM Update
    builder.add_edge("agent_Approved_Procedure", "agent_dB_Update")
    builder.add_edge("agent_Blocked_Procedure", "agent_dB_Update")
    builder.add_edge("agent_dB_Update", END)

    # Checkpointer Setup
    settings.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        settings.CHECKPOINT_DIR / settings.CHECKPOINT_DB,
        check_same_thread=False
    )
    memory = SqliteSaver(conn)

    return builder.compile(checkpointer=memory)


# Single pre-compiled workflow instance
#fraud_detection_workflow = build_fraud_detection_workflow()


#----------------------------------------------------------------------------------------
def init_checkpoint_db(force_reset:bool=False):
    """
    init and Create a new SQLite database for checkpoint by building the graph
    reset before creating if force_reset=True (also generate graph PNG)
    Args:
        - force_reset: if True, reset existing dB and then init/setup, otherwise no action is taken
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        db_path = settings.CHECKPOINT_DIR / settings.CHECKPOINT_DB
        if db_path.exists():
            if force_reset:
                db_path.unlink()
                print(f"Existing database at {db_path} removed.")
            else:
                #comment this print for less noisy output
                #print(f'database already exist at :{db_path} (no action is taken)')
                return
        else:
            print(f'No database at {db_path}, Init and setup for the first time!')

        # Ensure parent folder exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        fraud_detection_graph = build_fraud_detection_workflow()
        print("LanGraph Checkpoint Database created successfully.")

        # Also Generate graph PNG image
        path = settings.IMG_DIR / settings.WORKFLOW_GRAPH
        png_bytes = fraud_detection_graph.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)
        print(f"Graph successfully rendered and saved to {path}")
    except Exception as e:
        logger.error(f"Error in init_checkpoint_db at {db_path} with force_reset={force_reset}: {e}")
        raise


if __name__ == "__main__":
    init_checkpoint_db(force_reset=True)