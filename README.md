# AI-Powered Customer Support Automation System

A fully automated customer support system for ABC Technologies built using LangGraph, Llama3.2 (via Ollama), RAG pipeline, and SQLite memory.

---

## Project Structure
customer-support-agent/

├── main.py                  # Task 10: Entry point - interactive chat

├── requirements.txt         # Python dependencies

├── schema.sql               # Task 7: SQLite database schema

├── memory.db                # Auto-created SQLite memory database

├── src/

│   ├── init.py

│   ├── state.py             # Task 2: SupportState definition

│   ├── memory.py            # Task 7: SQLite memory management

│   ├── rag.py               # Task 6: RAG pipeline (TF-IDF retrieval)

│   ├── nodes.py             # Tasks 3,4,5,8,9: All LangGraph nodes

│   └── graph.py             # Task 1: LangGraph workflow

├── docs/                    # Task 6: Knowledge base documents

│   ├── company_policy.txt

│   ├── pricing_guide.txt

│   ├── technical_manual.txt

│   └── faq.txt

└── diagrams/

└── workflow_diagram.png # Task 1: LangGraph architecture diagram

---

## Tasks Completed

| Task | Description | File |
|------|-------------|------|
| Task 1 | LangGraph Workflow Design | `src/graph.py`, `diagrams/` |
| Task 2 | State Structure | `src/state.py` |
| Task 3 | Intent Classification Node | `src/nodes.py` |
| Task 4 | Conditional Routing | `src/graph.py`, `src/nodes.py` |
| Task 5 | Specialized Support Agents | `src/nodes.py` |
| Task 6 | RAG Pipeline | `src/rag.py`, `docs/` |
| Task 7 | SQLite Memory | `src/memory.py`, `schema.sql` |
| Task 8 | Human-in-the-Loop Approval | `src/nodes.py` |
| Task 9 | Supervisor Agent | `src/nodes.py` |
| Task 10 | Demo with 5 Sample Queries | `main.py` |

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- Ollama installed and running with llama3.2 model

### 1. Clone the Repository
```bash
git clone https://github.com/anshikaraj-hub/customer-support-agent.git
cd customer-support-agent
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install langchain-ollama
```

### 4. Make sure Ollama is running with llama3.2
```bash
ollama run llama3.2
```

### 5. Run the System
```bash
python main.py
```

---

## Demo Queries (Task 10)

Run these queries in order to demonstrate all system features:

| # | Query | Expected Path |
|---|-------|---------------|
| 1 | What are the pricing plans available for your software? | Sales Agent |
| 2 | I forgot my account password. | Account Agent |
| 3 | My application crashes whenever I upload a file. | Technical Agent |
| 4 | I need a refund for my annual subscription. | Billing Agent + Human Approval |
| 5 | What was my previous support issue? | Memory Recall |

---

## System Workflow
Customer Query

│

▼

Load Memory (SQLite)

│

▼

Intent Classifier (Llama3.2)

│

▼

RAG Retrieval (TF-IDF Knowledge Base)

│

▼

Conditional Router

│

┌────┼────┬────────┬────────┐

▼    ▼    ▼        ▼        ▼

Sales Tech Billing Account Memory

│

High-Risk?

┌─────┴─────┐

▼           ▼

Human         Skip

Approval

└─────┬─────┘

▼

Supervisor Agent

│

▼

Save Memory

│

▼

Final Response

---

## Human-in-the-Loop

The following request types require supervisor approval before a response is sent:
- Refund requests
- Subscription cancellations
- Account closure requests
- Compensation requests
- Escalation to management

---

## Knowledge Base Documents (RAG)

| Document | Covers |
|----------|--------|
| company_policy.txt | Refunds, cancellations, escalation policies |
| pricing_guide.txt | Plans, pricing, discounts, add-ons |
| technical_manual.txt | System requirements, error codes, troubleshooting |
| faq.txt | Common questions and answers |

---

## SQLite Memory Schema

```sql
CREATE TABLE conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   TEXT NOT NULL,
    customer_name TEXT,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    intent        TEXT,
    timestamp     TEXT NOT NULL
);
```

---

## Technologies Used

| Component | Technology |
|-----------|-----------|
| Workflow Orchestration | LangGraph |
| LLM | Llama3.2 via Ollama |
| RAG / Vector Search | TF-IDF (custom implementation) |
| Memory | SQLite3 |
| Language | Python 3.11 |