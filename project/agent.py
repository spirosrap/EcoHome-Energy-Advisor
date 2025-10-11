"""
Energy Advisor agent definition for EcoHome."""
from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ecohome_starter.tools import TOOL_KIT

load_dotenv()


class Agent:
    """EcoHome Energy Advisor agent built with LangGraph."""

    def __init__(self, instructions: str, model: str = "gpt-4o-mini") -> None:
        self.instructions = instructions
        self.model_name = model

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VOCAREUM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OpenAI credentials. Set OPENAI_API_KEY or VOCAREUM_API_KEY."
            )

        base_url = (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://openai.vocareum.com/v1"
        )

        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url=base_url,
            api_key=api_key,
        )

        self.tool_node = ToolNode(TOOL_KIT)
        self.graph = self._build_graph()

    def _llm_node(self, state: MessagesState) -> dict:
        """LLM node that appends the assistant response."""
        response = self.llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    @staticmethod
    def _route_next(state: MessagesState) -> str:
        """Determine whether to call tools or finish."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    def _build_graph(self):
        """Construct the LangGraph state graph with nodes and edges."""
        graph = StateGraph(MessagesState)
        graph.add_node("llm", self._llm_node)
        graph.add_node("tools", self.tool_node)

        graph.add_edge(START, "llm")
        graph.add_conditional_edges(
            "llm",
            self._route_next,
            path_map={"tools": "tools", END: END},
        )
        graph.add_edge("tools", "llm")
        graph.add_edge("llm", END)

        return graph.compile()

    def invoke(self, question: str, context: Optional[str] = None) -> dict:
        """Ask the agent a question with optional additional context."""
        messages: List = [SystemMessage(content=self.instructions)]
        if context:
            messages.append(SystemMessage(content=context))
        messages.append(HumanMessage(content=question))

        result = self.graph.invoke({"messages": messages})
        final_message = result["messages"][-1]
        return {
            "messages": result["messages"],
            "final_response": getattr(final_message, "content", str(final_message)),
        }

    def get_agent_tools(self) -> List[str]:
        """Return the names of available tools."""
        return [tool.name for tool in TOOL_KIT]
