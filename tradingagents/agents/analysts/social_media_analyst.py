from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from datetime import datetime, timedelta
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    ensure_tool_calls_attr,
    get_news,
    is_local_qwen_like,
    safe_tool_invoke,
)
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
        ]

        system_message = (
            "You are a social media and company specific news researcher/analyst tasked with analyzing social media posts, recent company news, and public sentiment for a specific company over the past week. You will be given a company's name your objective is to write a comprehensive long report detailing your analysis, insights, and implications for traders and investors on this company's current state after looking at social media and what people are saying about that company, analyzing sentiment data of what people feel each day about the company, and looking at recent company news. Use the get_news(query, start_date, end_date) tool to search for company-specific news and social media discussions. Try to look at all sources possible from social media to sentiment to news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
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
            end_date = current_date
            start_date = (
                datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=7)
            ).strftime("%Y-%m-%d")
            news_blob = safe_tool_invoke(
                get_news,
                {"ticker": ticker, "start_date": start_date, "end_date": end_date}
            )
            local_prompt = (
                f"{system_message}\n\n"
                "NOTE: You are running in LOCAL mode. The required tools have already "
                "been executed for you. Do NOT request tools; analyze the retrieved "
                "data below directly.\n\n"
                f"Ticker: {ticker}\nTrade date: {current_date}\n"
                f"Range: {start_date} -> {end_date}\n\n"
                f"## Retrieved company news / discussions\n{news_blob}"
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
