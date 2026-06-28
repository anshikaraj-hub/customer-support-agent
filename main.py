# Task 10: Demonstration
# Interactive customer support chat — user types queries manually.

from src.graph import graph

CUSTOMER_ID = "CUST-DEMO-001"


def run_query(query_text: str) -> dict:
    """Run a single query through the LangGraph workflow."""
    initial_state = {
        "customer_id":          CUSTOMER_ID,
        "customer_name":        None,
        "customer_query":       query_text,
        "intent":               None,
        "is_high_risk":         False,
        "retrieved_context":    None,
        "conversation_history": [],
        "requires_approval":    False,
        "approval_status":      None,
        "supervisor_notes":     None,
        "agent_response":       None,
        "final_response":       None,
        "department":           None,
        "error":                None,
    }
    return graph.invoke(initial_state)


def main():
    print("\n" + "=" * 65)
    print("  ABC Technologies - AI Customer Support Automation System")
    print("  Powered by LangGraph + Llama3.2 + RAG + SQLite Memory")
    print("=" * 65)
    print("\n  Type your query and press Enter.")
    print("  Type 'exit' to quit.\n")

    while True:
        print("─" * 65)
        query = input("  You: ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("\n  Thank you for using ABC Technologies Support. Goodbye!\n")
            break

        try:
            result = run_query(query)

            print(f"\n  [Routed to: {result.get('department')} Department]")
            if result.get('is_high_risk'):
                print(f"  [Approval Status: {result.get('approval_status')}]")
            print(f"\n  Support Agent:\n")
            print(f"  {result.get('final_response', '(No response generated)')}")
            print()

        except Exception as e:
            print(f"\n  ERROR: {e}\n")
            raise


if __name__ == "__main__":
    main()