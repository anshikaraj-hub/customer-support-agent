# Task 2: State Structure
# This defines the shared data object that flows through every node in the graph.

from typing import TypedDict, Optional, List


class SupportState(TypedDict):

    # Customer Information
    customer_id: str                    # Unique ID for the customer
    customer_name: Optional[str]        # Customer name (from memory or query)

    # Query Details
    customer_query: str                 # The raw message from the customer
    intent: Optional[str]              # Classified as: Sales, Technical, Billing, Account, Memory
    is_high_risk: bool                 # True if request needs human approval

    # RAG Context
    retrieved_context: Optional[str]   # Relevant text retrieved from knowledge base

    # Conversation Memory
    conversation_history: List[dict]   # Past messages: [{"role": ..., "content": ...}]

    # Human-in-the-Loop
    requires_approval: bool            # Flags high-risk requests for supervisor review
    approval_status: Optional[str]     # "pending", "approved", or "rejected"
    supervisor_notes: Optional[str]    # Notes added by the human supervisor

    # Response
    agent_response: Optional[str]      # Draft response from the specialized agent
    final_response: Optional[str]      # Final polished response sent to customer

    # Routing
    department: Optional[str]          # Which department handled this query
    error: Optional[str]               # Error message if something goes wrong