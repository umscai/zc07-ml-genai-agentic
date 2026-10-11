import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import settings
from orchestration.graph import build_fraud_detection_workflow
from agents.agent_common import get_init_state

logger = logging.getLogger(__name__)


class FraudDetectionAssistant:
    """Orchestrates transaction processing across persistent source-id threads."""

    def __init__(
        self,
        api_key: str = settings.OPENAI_API_KEY,
        model_name: str = settings.MODEL_NAME,
        temperature: float = settings.TEMPERATURE,
        base_url: str = settings.OPENAI_BASE_URL
    ):
        # Initialize LLM

        try:
            #ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=settings.MODEL_NAME)
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
                timeout=30.0,  # Total request timeout in seconds
                max_retries=2
            )
        except Exception as e:
            raise RuntimeError(f"LLM Initialization failed: {e}")

        # Build workflow graph (with checkpointer enabled)
        try:
            self.workflow = build_fraud_detection_workflow()
        except Exception as e:
            raise RuntimeError(f"Agent Graph creation failed: {e}")

    def run(self, source_id: str, transaction: Dict[str, float|int]) -> Dict[str, Any]:
        """
        Processes incoming transaction form a specific source.

        Uses source_id as thread_id so SqliteSaver automatically tracks history  per source.
        """
        # 1. Thread configuration mapping source_id -> thread_id
        config = {
            "configurable": {
                "thread_id": source_id,
                "llm": self.llm
            }
        }
        actual_class = transaction.pop('Class', 2) # only keep features

        # 2. Format transaction input into a HumanMessage
        user_msg = HumanMessage(content=f"transaction payload for {source_id}: {transaction}")

        # 3. Payload state input
        # Note: LangGraph checkpointer will merge this input with existing state for this thread_id

        # [NOTE] When unpacking dictionaries using {**dict1, "key": value},
        # if the same key appears multiple times, the rightmost (later) key always overwrites any earlier ones.
        input_state = {
            **get_init_state(),
            "source_id": source_id,
            "features": transaction,
            "messages": [user_msg],
            "actual_class": actual_class
        }
        features = input_state.get("features") or {}

        if not features:  # Checks for None, empty dict {}, or empty string
            print("Warning: 'features' is empty. Skipping graph execution.")
            # Return early or handle fallback response here
            return {
                "success": False,
                "response": "Empty features skipped, graph execution",
                "state": {},
                "error": 'Empty Feature/Transaction'
            }

        try:
            # Invoke graph with checkpointer configuration
            ## SqliteSaver handles thread lookup, state re-hydration, and merging automatically (benefit of using persistence checkpointer instead of RAM)
            final_state = self.workflow.invoke(input_state, config=config)
            final_resp_dic = final_state.get("response_dic", {})

            return {
                "success": True,
                "response_dic": final_resp_dic,
                "state": final_state,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error in FraudDetectionAssistant.run for {source_id}: [{type(e).__name__}] {str(e) or repr(e)}", exc_info=True)
            #logger.error(f"Error in FraudDetectionAssistant.run for {source_id}: {str(e)}")
            return {
                "success": False,
                "response_dic": {},
                "state": {},
                "error": str(e)
            }


