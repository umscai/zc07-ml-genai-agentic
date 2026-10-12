from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine, inspect

from config import settings
from orchestration.graph import build_fraud_detection_workflow
import logging
logger = logging.getLogger(__name__)
from matplotlib.axes import Axes # for type-hint


def add_percentage_labels_barh(ax:Axes, format_str="{:.1f}%"):
    """
    Helper function to add percentage labels to the right side of horizontal bars,
    normalized to the maximum value in the plot.

    Args:
    - ax: matplotlib Axes object
    - format_str: string format for the percentage (default one decimal place)
    Returns:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        # Get the maximum width among all bars to use as the 100% baseline
        patches = ax.patches
        if not patches:
            return

        max_width = max(patch.get_width() for patch in patches)
        if max_width == 0:
            return

        # Small offset to prevent text from touching the end of the bar
        offset = max_width * 0.015

        for patch in patches:
            width = patch.get_width()
            if width <= 0:
                continue

            # Calculate percentage relative to the max value
            percentage = (width / max_width) * 100
            label_text = format_str.format(percentage)

            # Get the vertical center of the bar
            y_pos = patch.get_y() + patch.get_height() / 2.0

            # Place text slightly to the right of the bar
            ax.text(
                width + offset,
                y_pos,
                label_text,
                va='center',
                ha='left',
                fontsize=9,
                color='black',
                weight='bold'
            )

        # Extend the x-axis slightly so the labels aren't cut off at the edge
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0], xlim[1] * 1.15)
    except Exception as e:
        logger.error(f'Error in add_percentage_labels_barh: {e}')


class GetWorkflowAnalytics:
    """Get and plot token usage, tool usage, func usage, and confidence scores of Assets."""

    def __init__(self, checkpoint_db_path: Path = settings.CHECKPOINT_DIR / settings.CHECKPOINT_DB):
        self.graph = build_fraud_detection_workflow()
        self.engine = create_engine(f"sqlite:///{checkpoint_db_path}", echo=False)
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()
        self.table = "checkpoints" if "checkpoints" in table_names else None
        self.df_usage = None

    def get_source_ids(self):
        if self.table:
            df = pd.read_sql_table(self.table, self.engine)
            sids = df["thread_id"].unique()
            return sids
        return None

    def get_token_usage(self, msg_list: list) -> dict:
        """
        Extracts a token usage in the message history.
        Args:
            msg_list(list): list of LangChain/LangGraph-formatted messages
        Returns:
            A dict contain input, output and total usage
        """
        results = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for msg in msg_list:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                for k in results:
                    results[k] += msg.usage_metadata.get(k, 0)
        return results

    def get_tool_usage(self, msg_list: list) -> list:
        """
        Extracts a list of all tool names invoked in the message history.
        Args:
            msg_list(list): list of LangChain/LangGraph-formatted messages
        Returns:
            A list, preserving order and allowing for multiple calls to the same tool.
        """
        tool_names = []
        for msg in msg_list:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    name = tool_call.get("name")
                    if name:
                        tool_names.append(name)
        return tool_names

    def get_usage_per_source(self):
        """ Get usage per source from Checkpoint dB"""
        sids = self.get_source_ids()
        if sids is None:
            return None

        usage_data = []
        for sid in sids:
            config = {"configurable": {"thread_id": sid}}
            state = self.graph.get_state(config)

            # Extract state values
            messages = state.values.get("messages", [])
            #confidence = state.values.get("confidence")
            func_used = state.values.get("func_used") or []
            node_called = state.values.get("node_called") or []

            # Get Token Metrics & Tool Usage
            tokens = self.get_token_usage(messages)
            tools_called = self.get_tool_usage(messages)

            # Combine data
            entry = {
                "source_id": sid,
                "tools_used": tools_called,
                "tool_count": len(tools_called),
                "func_used": func_used,
                "func_count": len(func_used),
                "node_called": node_called,
                "node_count": len(node_called),
                #"confidence": confidence,
                **tokens,
            }

            if tokens["total_tokens"] > 0 or len(tools_called) > 0 or len(func_used) > 0:
                usage_data.append(entry)

        self.df_usage = pd.DataFrame(usage_data)
        return self.df_usage

    def get_df_usage(self):
        if self.df_usage is None:
            return self.get_usage_per_source()
        return self.df_usage

    def plot_token_usage_per_sid(self):
        """Plot Token usage by AI/LLM."""
        df_usage = self.get_df_usage()
        df_usage.plot(
            x="source_id",
            y=["input_tokens", "output_tokens"],
            kind="bar",
            stacked=True,
            figsize=(10, 6),
            color=['tab:blue', 'tab:red'], #["#3498db", "#e74c3c"],
            rot=0,
        )
        plt.title("Token Usage Breakdown per Source")
        plt.ylabel("Token Count")
        plt.xlabel("Source ID")
        plt.tight_layout()
        plt.savefig(settings.IMG_DIR / "token_usage.png", dpi=300)
        plt.show()

    def plot_tool_usage_count(self):
        """Plot frequency of workflow tool usage by AI/LLM."""
        df_usage = self.get_df_usage()
        all_tools = [tool for sublist in df_usage["tools_used"] for tool in sublist]
        tool_counts = pd.Series(all_tools).value_counts(sort=True, ascending=True)

        ax=tool_counts.plot(kind="barh", color="tab:green", figsize=(8, 5)) #color="#2ecc71"
        add_percentage_labels_barh(ax)
        plt.title("AI-Agent Tool Usage Count")
        plt.xlabel("Total Number of Invocations")
        plt.tight_layout()
        plt.savefig(settings.IMG_DIR / "tool_usage.png", dpi=300)
        plt.show()

    def plot_func_usage_count(self):
        """Plot frequency of workflow functions used."""
        df_usage = self.get_df_usage()
        all_funcs = [func for sublist in df_usage["func_used"] for func in sublist]

        if not all_funcs:
            print("No function usage data available to plot.")
            return

        func_counts = pd.Series(all_funcs).value_counts(sort=True, ascending=True)

        ax=func_counts.plot(kind="barh", color="tab:purple", figsize=(8, 5)) #color="#9b59b6"
        add_percentage_labels_barh(ax)
        plt.title("Node Functions Usage Count")
        plt.xlabel("Total Number of Executions")
        plt.tight_layout()
        plt.savefig(settings.IMG_DIR / "func_usage.png", dpi=300)
        plt.show()

    def plot_node_usage_count(self):
        """Plot frequency of workflow node used."""
        df_usage = self.get_df_usage()
        all_nodes = [node for sublist in df_usage["node_called"] for node in sublist]

        if not all_nodes:
            print("No node call data available to plot.")
            return

        node_counts = pd.Series(all_nodes).value_counts(sort=True, ascending=True)

        node_counts_df=node_counts.to_frame(name='usage_count')
        node_counts_df.index.name = 'node'
        node_counts_df.to_csv(settings.IMG_DIR / "node_usage.csv")

        ax = node_counts.plot(kind="barh", color="tab:orange", figsize=(8, 5)) #color="#9b59b6
        add_percentage_labels_barh(ax)
        plt.title("Node Usage Count")
        plt.xlabel("Total Number of Executions")
        plt.tight_layout()
        plt.savefig(settings.IMG_DIR / "node_usage.png", dpi=300)
        plt.show()
        return node_counts_df

    # def plot_confidence_distribution(self):
    #     """Plot histogram with KDE for state confidence scores."""
    #     df_usage = self.get_df_usage()
    #     df_confidence = df_usage["confidence"].dropna()

    #     if df_confidence.empty:
    #         print("No confidence score data available to plot.")
    #         return

    #     plt.figure(figsize=(8, 5))
    #     sns.histplot(df_confidence, kde=True, color="#e67e22", bins=10)
    #     plt.title("Confidence Score Distribution")
    #     plt.xlabel("Confidence")
    #     plt.ylabel("Density / Count")
    #     plt.xlim(0, 1.0)  # Adjust range if confidence isn't bounded [0, 1]
    #     plt.tight_layout()
    #     plt.savefig(settings.IMG_DIR / "confidence_distribution.png", dpi=300)
    #     plt.show()


def get_checkpoint_df()->pd.DataFrame|None:
    """
    get checkpoint-dB data in data-frame format
    Args:
        - None
    Returns:
        pandas dataframe contains telemetry log
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        db_path = settings.CHECKPOINT_DIR / settings.CHECKPOINT_DB
        if db_path.exists():
            table_name = 'checkpoints'
            engine = create_engine(f"sqlite:///{settings.CHECKPOINT_DIR / settings.CHECKPOINT_DB}", echo=False)
            df = pd.read_sql_table(table_name, engine)
            return df
        else:
            return None
    except Exception as e:
        logger.error(f"Error in get_ltm_df: {e}")
        raise

if __name__ == "__main__":
    analytics = GetWorkflowAnalytics()
    analytics.plot_token_usage_per_sid()
    analytics.plot_tool_usage_count()
    analytics.plot_func_usage_count()
    analytics.plot_node_usage_count()
    #analytics.plot_confidence_distribution()