from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )


def is_local_qwen_like(llm) -> bool:
    """Return True for the LocalQwen adapter or a thin wrapper around it."""
    if llm.__class__.__name__ == "LocalQwenChat":
        return True
    inner = getattr(llm, "inner", None)
    return inner is not None and inner.__class__.__name__ == "LocalQwenChat"


def ensure_tool_calls_attr(message) -> None:
    """Keep graph routing safe when a local-path message has no tool calls."""
    if hasattr(message, "tool_calls") and getattr(message, "tool_calls", None) is not None:
        return
    try:
        message.tool_calls = []  # type: ignore[attr-defined]
    except Exception:
        setattr(message, "tool_calls", [])


def safe_tool_invoke(tool, payload: dict) -> str:
    """Run a tool and surface failures as text so local-mode analysis can continue."""
    try:
        return tool.invoke(payload)
    except Exception as exc:
        return f"ERROR running {tool.name}: {exc}"

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
