# Tasks 3, 4, 5, 8, 9: All LangGraph Nodes
# - Task 3: Intent Classification
# - Task 4: Conditional Routing
# - Task 5: Specialized Support Agents (Sales, Technical, Billing, Account, Memory)
# - Task 8: Human-in-the-Loop Approval
# - Task 9: Supervisor Agent

import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import SupportState
from src.memory import (
    get_conversation_history,
    get_customer_name,
    save_message,
    format_history_for_prompt,
)
from src.rag import retrieve_context

# Load the local Ollama model
llm = ChatOllama(model="llama3.2", temperature=0.3)

# Keywords that trigger human approval (Task 8)
HIGH_RISK_KEYWORDS = [
    "refund",
    "cancel subscription",
    "cancellation",
    "account closure",
    "close account",
    "compensation",
    "escalate",
    "speak to manager",
    "escalation to management",
    "delete account",
]


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Helper function to call the local Ollama LLM."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ─────────────────────────────────────────────
# NODE: Load Memory (Task 7)
# ─────────────────────────────────────────────
def load_memory_node(state: SupportState) -> SupportState:
    """Load customer conversation history from SQLite."""
    print(f"\n[Memory] Loading history for customer: {state['customer_id']}")

    history = get_conversation_history(state["customer_id"])
    name = get_customer_name(state["customer_id"]) or state.get("customer_name")

    return {
        **state,
        "conversation_history": history,
        "customer_name": name,
    }


# ─────────────────────────────────────────────
# NODE: Intent Classification (Task 3)
# ─────────────────────────────────────────────
def classify_intent_node(state: SupportState) -> SupportState:
    """Classify the customer query into the correct department."""
    print(f"\n[Intent] Classifying: {state['customer_query'][:60]}...")

    query = state["customer_query"]

    # Check for memory recall queries first
    memory_keywords = [
        "previous issue", "last query", "what did i ask",
        "before", "earlier", "my history", "previous support"
    ]
    if any(kw in query.lower() for kw in memory_keywords):
        print("[Intent] -> Memory recall detected")
        return {**state, "intent": "Memory", "department": "Memory"}

    system = """You are an intent classifier for a customer support system.
Classify the customer query into EXACTLY one of these categories:
- Sales: pricing, plans, subscriptions, features, product information
- Technical: app errors, crashes, login issues, installation, configuration
- Billing: invoices, payments, refunds, billing issues
- Account: password reset, profile updates, account activation or deactivation

Reply with ONLY the category name. Nothing else."""

    user = f"Customer query: {query}"

    intent = call_llm(system, user)

    # Normalize the response to one of the valid intents
    for valid in ["Sales", "Technical", "Billing", "Account"]:
        if valid.lower() in intent.lower():
            intent = valid
            break
    else:
        intent = "Sales"

    print(f"[Intent] -> {intent}")
    return {**state, "intent": intent, "department": intent}


# ─────────────────────────────────────────────
# NODE: RAG Retrieval (Task 6)
# ─────────────────────────────────────────────
def rag_retrieval_node(state: SupportState) -> SupportState:
    """Retrieve relevant context from the knowledge base."""
    print(f"\n[RAG] Retrieving context...")

    context = retrieve_context(state["customer_query"], n_results=3)
    print(f"[RAG] Retrieved {len(context)} characters of context.")
    return {**state, "retrieved_context": context}


# ─────────────────────────────────────────────
# ROUTING FUNCTION (Task 4)
# ─────────────────────────────────────────────
def route_to_department(state: SupportState) -> str:
    """Conditional routing — directs query to the correct agent node."""
    intent = state.get("intent", "Sales")
    route_map = {
        "Sales":     "sales_agent",
        "Technical": "technical_agent",
        "Billing":   "billing_agent",
        "Account":   "account_agent",
        "Memory":    "memory_agent",
    }
    destination = route_map.get(intent, "sales_agent")
    print(f"\n[Router] Routing to: {destination}")
    return destination


# ─────────────────────────────────────────────
# HELPER: Build Agent Response (Task 5)
# ─────────────────────────────────────────────
def _build_agent_response(department: str, role_desc: str, state: SupportState) -> SupportState:
    """Shared helper used by all specialized agents."""
    query = state["customer_query"]
    context = state.get("retrieved_context", "")
    history_text = format_history_for_prompt(state.get("conversation_history", []))
    name = state.get("customer_name", "Customer")

    system = f"""You are a {department} Support Agent for ABC Technologies.
{role_desc}
Use the knowledge base context and conversation history to answer accurately.
Be professional, empathetic, and concise.
Address the customer by name if you know it.
Keep your response under 150 words."""

    user = f"""Customer Name: {name}
Customer Query: {query}

Knowledge Base Context:
{context}

Previous Conversation:
{history_text}

Provide a helpful response:"""

    response = call_llm(system, user)

    # Check if this request is high risk
    is_high_risk = any(kw in query.lower() for kw in HIGH_RISK_KEYWORDS)
    requires_approval = is_high_risk and department == "Billing"

    print(f"[{department} Agent] Response generated. High-risk: {is_high_risk}")

    return {
        **state,
        "agent_response": response,
        "is_high_risk": is_high_risk,
        "requires_approval": requires_approval,
        "approval_status": "pending" if requires_approval else None,
    }


# ─────────────────────────────────────────────
# SPECIALIZED AGENTS (Task 5)
# ─────────────────────────────────────────────
def sales_agent_node(state: SupportState) -> SupportState:
    """Handles pricing, plans, and product information queries."""
    print("\n[Sales Agent] Handling query...")
    return _build_agent_response(
        "Sales",
        "You handle product information, subscription plans, pricing details, and feature inquiries.",
        state,
    )


def technical_agent_node(state: SupportState) -> SupportState:
    """Handles application errors, crashes, and configuration issues."""
    print("\n[Technical Agent] Handling query...")
    return _build_agent_response(
        "Technical",
        "You handle application errors, crashes, login problems, installation, and configuration issues.",
        state,
    )


def billing_agent_node(state: SupportState) -> SupportState:
    """Handles invoices, payments, and refund requests."""
    print("\n[Billing Agent] Handling query...")
    return _build_agent_response(
        "Billing",
        "You handle invoice requests, payment issues, refund requests, and billing disputes.",
        state,
    )


def account_agent_node(state: SupportState) -> SupportState:
    """Handles password resets, profile updates, and account management."""
    print("\n[Account Agent] Handling query...")
    return _build_agent_response(
        "Account",
        "You handle password resets, profile updates, account activation, and account deactivation.",
        state,
    )


def memory_agent_node(state: SupportState) -> SupportState:
    """Handles queries about previous interactions using stored memory."""
    print("\n[Memory Agent] Handling memory recall query...")

    history_text = format_history_for_prompt(state.get("conversation_history", []))
    name = state.get("customer_name", "Customer")

    system = """You are a helpful customer support assistant.
The customer is asking about their previous support interactions.
Use the conversation history below to answer their question accurately.
If no history exists, politely let them know."""

    user = f"""Customer Name: {name}
Customer Query: {state['customer_query']}

Previous Conversation History:
{history_text}

Answer the customer's question:"""

    response = call_llm(system, user)

    return {
        **state,
        "agent_response": response,
        "is_high_risk": False,
        "requires_approval": False,
        "approval_status": None,
    }


# ─────────────────────────────────────────────
# NODE: Human-in-the-Loop Approval (Task 8)
# ─────────────────────────────────────────────
def human_approval_node(state: SupportState) -> SupportState:
    """Pauses workflow and asks a human supervisor to approve or reject high-risk requests."""
    print("\n" + "=" * 60)
    print("[Human Approval] HIGH-RISK REQUEST DETECTED")
    print("=" * 60)
    print(f"Customer  : {state.get('customer_name', state['customer_id'])}")
    print(f"Query     : {state['customer_query']}")
    print(f"Department: {state.get('department')}")
    print(f"\nDraft Response:\n{state.get('agent_response')}")
    print("=" * 60)

    decision = input("\nSupervisor: Approve this response? (yes/no): ").strip().lower()

    if decision in ["yes", "y", ""]:
        notes = input("Supervisor notes (optional, press Enter to skip): ").strip()
        print("[Human Approval] Request APPROVED.")
        return {
            **state,
            "approval_status": "approved",
            "supervisor_notes": notes or "Approved by supervisor.",
        }
    else:
        reason = input("Supervisor: Reason for rejection: ").strip()
        print("[Human Approval] Request REJECTED.")
        return {
            **state,
            "approval_status": "rejected",
            "supervisor_notes": reason or "Rejected by supervisor.",
            "agent_response": (
                f"Your request has been escalated to our team. "
                f"A supervisor will contact you within 24 hours. "
                f"Reason: {reason or 'Under review'}."
            ),
        }


def check_approval_needed(state: SupportState) -> str:
    """Edge function — checks whether human approval is required."""
    if state.get("requires_approval"):
        print("[Router] High-risk request detected. Routing to human approval.")
        return "needs_approval"
    return "no_approval"


# ─────────────────────────────────────────────
# NODE: Supervisor Agent (Task 9)
# ─────────────────────────────────────────────
def supervisor_agent_node(state: SupportState) -> SupportState:
    """Validates and polishes the agent response before sending to the customer."""
    print("\n[Supervisor Agent] Validating response quality...")

    draft = state.get("agent_response", "")
    query = state["customer_query"]
    context = state.get("retrieved_context", "")
    approval_status = state.get("approval_status", "")
    supervisor_notes = state.get("supervisor_notes", "")

    system = """You are a Quality Assurance Supervisor for ABC Technologies customer support.
Review the draft response and improve it if needed.
Make sure it:
1. Directly answers the customer query
2. Is professional and empathetic
3. Is accurate based on the knowledge base context
4. Is concise and under 150 words
5. Has a clear next step or call to action if needed
6. Does not reveal any internal system details

If the draft is already good, return it with minor polish only."""

    user = f"""Customer Query: {query}

Knowledge Base Context:
{context}

Draft Response:
{draft}

Supervisor Notes: {supervisor_notes if supervisor_notes else 'None'}
Approval Status: {approval_status if approval_status else 'Not required'}

Provide the final polished response:"""

    final = call_llm(system, user)
    print("[Supervisor Agent] Final response ready.")
    return {**state, "final_response": final}


# ─────────────────────────────────────────────
# NODE: Save Memory (Task 7)
# ─────────────────────────────────────────────
def save_memory_node(state: SupportState) -> SupportState:
    """Save the current interaction to SQLite for future reference."""
    print("\n[Memory] Saving conversation to SQLite...")

    name = state.get("customer_name")
    if not name:
        match = re.search(r"my name is (\w+)", state["customer_query"], re.IGNORECASE)
        if match:
            name = match.group(1)

    save_message(
        customer_id=state["customer_id"],
        role="user",
        content=state["customer_query"],
        customer_name=name,
        intent=state.get("intent"),
    )
    save_message(
        customer_id=state["customer_id"],
        role="assistant",
        content=state.get("final_response", ""),
        customer_name=name,
        intent=state.get("intent"),
    )

    print("[Memory] Conversation saved successfully.")
    return {**state, "customer_name": name or state.get("customer_name")}