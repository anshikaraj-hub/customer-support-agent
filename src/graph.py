# Task 1: LangGraph Workflow Design
# Task 4: Conditional Routing
# This file builds and compiles the full LangGraph workflow.

from langgraph.graph import StateGraph, END

from src.state import SupportState
from src.nodes import (
    load_memory_node,
    classify_intent_node,
    rag_retrieval_node,
    route_to_department,
    sales_agent_node,
    technical_agent_node,
    billing_agent_node,
    account_agent_node,
    memory_agent_node,
    human_approval_node,
    check_approval_needed,
    supervisor_agent_node,
    save_memory_node,
)


def build_graph():
    """
    Build the full LangGraph workflow.

    Flow:
    load_memory
        -> classify_intent
            -> rag_retrieval
                -> [route_to_department]
                    -> sales_agent
                    -> technical_agent
                    -> billing_agent
                    -> account_agent
                    -> memory_agent
                -> [check_approval_needed]
                    -> human_approval (if high-risk)
                    -> supervisor_agent (if standard)
                -> supervisor_agent
                    -> save_memory
                        -> END
    """

    workflow = StateGraph(SupportState)

    # ── Add all nodes ──
    workflow.add_node("load_memory",       load_memory_node)
    workflow.add_node("classify_intent",   classify_intent_node)
    workflow.add_node("rag_retrieval",     rag_retrieval_node)
    workflow.add_node("sales_agent",       sales_agent_node)
    workflow.add_node("technical_agent",   technical_agent_node)
    workflow.add_node("billing_agent",     billing_agent_node)
    workflow.add_node("account_agent",     account_agent_node)
    workflow.add_node("memory_agent",      memory_agent_node)
    workflow.add_node("human_approval",    human_approval_node)
    workflow.add_node("supervisor_agent",  supervisor_agent_node)
    workflow.add_node("save_memory",       save_memory_node)

    # ── Entry point ──
    workflow.set_entry_point("load_memory")

    # ── Linear edges ──
    workflow.add_edge("load_memory",     "classify_intent")
    workflow.add_edge("classify_intent", "rag_retrieval")

    # ── Conditional routing: RAG -> correct agent (Task 4) ──
    workflow.add_conditional_edges(
        "rag_retrieval",
        route_to_department,
        {
            "sales_agent":     "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent":   "billing_agent",
            "account_agent":   "account_agent",
            "memory_agent":    "memory_agent",
        }
    )

    # ── All agents go to approval check ──
    for agent in [
        "sales_agent",
        "technical_agent",
        "billing_agent",
        "account_agent",
        "memory_agent",
    ]:
        workflow.add_conditional_edges(
            agent,
            check_approval_needed,
            {
                "needs_approval": "human_approval",
                "no_approval":    "supervisor_agent",
            }
        )

    # ── Human approval always goes to supervisor ──
    workflow.add_edge("human_approval",   "supervisor_agent")

    # ── Supervisor saves to memory then ends ──
    workflow.add_edge("supervisor_agent", "save_memory")
    workflow.add_edge("save_memory",      END)

    return workflow.compile()


# Compile once when imported
graph = build_graph()