# Task 10: Demonstration
# Run all 5 sample queries to demonstrate the complete system.

import os
import sys

from src.graph import graph

# ── 5 Sample Queries from the Assignment ──
DEMO_QUERIES = [
    {
        "id": 1,
        "query": "What are the pricing plans available for your software?",
        "expected": "Sales",
    },
    {
        "id": 2,
        "query": "I forgot my account password.",
        "expected": "Account",
    },
    {
        "id": 3,
        "query": "My application crashes whenever I upload a file.",
        "expected": "Technical Support",
    },
    {
        "id": 4,
        "query": "I need a refund for my annual subscription.",
        "expected": "Billing - requires human approval",
    },
    {
        "id": 5,
        "query": "What was my previous support issue?",
        "expected": "Memory recall",
    },
]

# Use a fixed customer ID so memory works across queries
CUSTOMER_ID = "CUST-DEMO-001"


def run_query(query_text: str) -> dict:
    """Run a single query through the LangGraph workflow."""
    initial_state = {
        "customer_id":           CUSTOMER_ID,
        "customer_name":         None,
        "customer_query":        query_text,
        "intent":                None,
        "is_high_risk":          False,
        "retrieved_context":     None,
        "conversation_history":  [],
        "requires_approval":     False,
        "approval_status":       None,
        "supervisor_notes":      None,
        "agent_response":        None,
        "final_response":        None,
        "department":            None,
        "error":                 None,
    }
    return graph.invoke(initial_state)


def run_demo():
    """Run all 5 demonstration queries one by one."""
    print("\n" + "=" * 65)
    print("  ABC Technologies - AI Customer Support Automation System")
    print("  Powered by LangGraph + Llama3.2 + RAG + SQLite Memory")
    print("=" * 65)

    for demo in DEMO_QUERIES:
        print(f"\n{'=' * 65}")
        print(f"  Query {demo['id']} | Expected Path: {demo['expected']}")
        print(f"{'=' * 65}")
        print(f"\n  Customer: {demo['query']}\n")

        try:
            result = run_query(demo["query"])

            print(f"\n{'─' * 65}")
            print(f"  RESULT SUMMARY")
            print(f"{'─' * 65}")
            print(f"  Intent     : {result.get('intent')}")
            print(f"  Department : {result.get('department')}")
            print(f"  High Risk  : {result.get('is_high_risk')}")
            print(f"  Approval   : {result.get('approval_status', 'Not required')}")
            print(f"\n  Final Response:\n")
            print(f"  {result.get('final_response', '(No response generated)')}")
            print(f"{'─' * 65}")

        except Exception as e:
            print(f"\n  ERROR: {e}")
            raise

        if demo["id"] < 5:
            input("\n  Press Enter to continue to the next query...\n")

    print(f"\n{'=' * 65}")
    print("  All 5 queries completed successfully!")
    print("  Memory stored in: memory.db")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    run_demo()