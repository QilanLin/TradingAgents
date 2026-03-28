from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime, timedelta
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    ensure_tool_calls_attr,
    get_global_news,
    get_news,
    is_local_qwen_like,
    safe_tool_invoke,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        if is_local_qwen_like(llm):
            global_blob = safe_tool_invoke(
                get_global_news,
                {"curr_date": current_date, "look_back_days": 7, "limit": 12}
            )
            end_date = current_date
            start_date = (
                datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=7)
            ).strftime("%Y-%m-%d")
            company_blob = safe_tool_invoke(
                get_news,
                {"ticker": ticker, "start_date": start_date, "end_date": end_date}
            )
            local_prompt = (
                f"{system_message}\n\n"
                "NOTE: You are running in LOCAL mode. The required tools have already "
                "been executed for you. Do NOT request tools; analyze the retrieved "
                "data below directly.\n\n"
                f"Ticker: {ticker}\nTrade date: {current_date}\n\n"
                f"## Global / macro news (last 7 days)\n{global_blob}\n\n"
                f"## Company-specific news (last 7 days)\n{company_blob}"
            )
            result = llm.invoke(local_prompt)
            report = getattr(result, "content", "")
            ensure_tool_calls_attr(result)
        else:
            chain = prompt | llm.bind_tools(tools)
            result = chain.invoke(state["messages"])
            report = ""
            if len(result.tool_calls) == 0:
                report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
