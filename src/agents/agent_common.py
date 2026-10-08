from typing import TypedDict, Annotated, List, Dict, Any, Tuple, Optional, Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages # or we could use from langgraph.graph.message import MessagesState
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
import json
from typing import Any
import operator
import logging
from dataclasses import asdict, dataclass
logger = logging.getLogger(__name__)

# Define allowed severity band values
SeverityBand = Literal["Critically_Low", "Low", "Normal", "High", "Critically_High"]

@dataclass
class GraphNode:
    start: str = 'START'
    end: str = 'END'
    detector: str = 'Detector'
    llm: str ='LLM_Agent'
    human: str = 'Human_Agent'
    dB: str = 'Update_dB'
    approved: str = 'Approved_Procedure'
    blocked: str = 'Blocked_Procedure'

@dataclass
class GraphAction:
    escalate: str = 'Escalate'
    #classified: str = 'Classified'
    approved: str = 'APPROVED'
    blocked: str = 'BLOCKED'
    in_prog:str ="In_Progress"

# Define the state object
class AgentState(TypedDict):
    # LangGraph Message History (mainly used for auditing)
    messages: Annotated[list[BaseMessage], add_messages] # or remove this line and use class AgentState(MessagesState):

    source_id:Literal['real','gen', 'unknown'] # if the transaction is real or generated data (later can be customer id)

    # Transaction info
    features: dict[str, float|int] # V1 to V28, Amount, and HourT=Hour/2

    # Actual Class, we have this because we have controlled generated data (otherwise will be determined by post verification)
    actual_class: Literal [0, 1, 2]  # 0=Legit, 1=Fraud, 2=unknown

    #predicted class: final prediction by Detector or Agent, or Human
    pred_class: Literal[0, 1, 2]

    fraud_prob:float # fraud probability determined by ML-Classifier

    #agentic workflow related info
    #LTM(Long-Term Memory)-related info this will be Vector database
    context: str # Long-Term Memory

    next_node: GraphNode | None
    current_node: GraphNode | None
    decision_node: GraphNode | None # The node which really make decision as Legit/Fraud(or we could add previous_node, but this will avoid any ambiguity)
    action: GraphAction

    response_dic:dict # short report from Detector, LLM, and Human in dict-format

    tool_used: List[str] | None  # current tool used by LLM-Agent, History will be part of messages

    # Use operator.add so node outputs append to the list instead of replacing it (we need this to keep it as history not part of messages)
    func_used: Annotated[List[str], operator.add]

    # logging timestamp
    timestamp_str:str

    #
    error: str | None # in case of node error



def get_init_state() -> dict:
    return {
        #'messages': [], # We may not need to re-init with empty list it has operator.add
        'source_id': 'unknown',
        'features': {},
        'actual_class': 2,
        'pred_class': 2,
        'fraud_prob': 1.0,
        'context': '',
        'next_node': None,
        'current_node': None,
        'decision_node': None,
        'action': 'In_Progress',
        'response_dic': {},
        'tool_used': [],
        #'func_used': [], # We may not need to re-init with empty list it has operator.add
        'timestamp_str': '',
        'error': None
    }

# Usage: Safely unpacking a brand-new instance
# input_state = {
#     **get_init_state(), # If we use static definition we may need to use **copy.deepcopy(INIT_STATE),otherwise can be a problem
#     "source_id": source_id,
#     "features": transaction,
#     "messages": [user_msg]
# }

def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> Tuple[Dict[str, Any], List[str]]:
    """
    Create react agent based on the given models, and output schema with tools, and return both results and tool_used
    Args:
        response_schema(type[BaseModel]): pydantic schema of the output
        messages(List[BaseMessage]): list of LangChain messages
        llm: LLM to use for creating agent
        tools: tools that can be bind and used by the agent
    Returns:
        return both agent response and the tool used
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        llm_with_tools = llm.bind_tools(tools)
        agent= create_agent(
            model=llm_with_tools,  # Use the bound model
            tools=tools,
            response_format=response_schema,
            )

        result = agent.invoke({"messages": messages})
        tools_used = [t.name for t in result.get("messages", []) if isinstance(t, ToolMessage)]

        return result, tools_used
    except Exception as e:
        logger.error(f"Error in invoke_react_agent: {e}")
        raise
