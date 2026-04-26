streamlit run app.py --server.fileWatcherType none

<div align="center">

# ⚖️ Multi-Agent Debate AI System

### _Intelligent Decision Making through Structured Agent Debate_

<br/>

[🚀 Live Demo](#-live-demo) • [✨ Features](#-features) • [🏗️ Architecture](#️-architecture) • [⚡ Quick Start](#-quick-start) • [📁 Project Structure](#-project-structure) • [🤝 Contributing](#-contributing)

</div>

---

## 🎯 What is this?

This system simulates **expert panel discussions** using 8 specialised LLM-powered agents. Each agent has a unique role, personality, and reasoning style. They argue over 3 structured rounds, challenge each other's claims, verify facts, and collectively produce a **scored, explainable final decision**.

Think of it as an AI-powered board of directors debating your business question — and delivering a verdict.

```
You ask: "Should our startup pivot from B2C to B2B?"

  🗂️  Planner     → Structures the debate into 4 focused propositions
  🌟  Optimist    → Champions the pivot with market opportunity data
  🔍  Skeptic     → Challenges assumptions, highlights execution risks
  📊  Analyst     → Grounds arguments in unit economics and growth metrics
  🎓  Expert      → Provides McKinsey-level strategic depth
  ⚡  Critic      → Scores every argument on 4 quality metrics
  ✅  Fact-Checker → Flags hallucinations and verifies claims
  ⚖️  Judge       → Delivers final decision with confidence score & risk level
```

---

## ✨ Features

### 🤖 Multi-Agent Debate Engine

- **8 specialised agents** with distinct personas, reasoning styles, and temperature settings
- **3-round structured debate** — Opening → Counterargument → Closing
- Agents reference each other's prior arguments via debate history injection
- Domain-aware prompts that adapt agent behaviour per field

### 🔍 Retrieval-Augmented Generation (RAG)

- Upload **PDF or TXT documents** for evidence-grounded debates
- Local embedding with **SentenceTransformers** (all-MiniLM-L6-v2, free, offline)
- **FAISS vector store** for sub-millisecond similarity search
- Retrieved chunks injected into every agent's context

### 📊 Scoring System

- Critic scores every agent on **4 metrics × 3 rounds = 12 data points per agent**
  - Logical Consistency, Evidence Usage, Relevance, Persuasiveness
- **Weighted scoring** — later rounds count more (Round 3 = 40%)
- Ranked leaderboard with debate winner declared

### 🧠 Memory System

- **Short-term memory** — live session state during debate
- **Long-term memory** — FAISS-indexed debate history for semantic recall
- Search past debates by topic similarity
- Auto-rotation at 100 stored debates

### ⚖️ Judge & Verdict

- Final decision with **confidence score (0-100%)**
- Risk level assessment (Low / Medium / High / Critical)
- Key reasons + acknowledged dissenting view
- Full reasoning explanation

### 📈 Analytics Dashboard

- Agent leaderboard with progress bars
- Per-metric breakdown bar charts
- Round-by-round score progression line chart
- Token usage per agent
- Fact-check accuracy summary

### 🌐 Multi-Domain Support

| Domain               | Agent Persona       | Key Metrics               |
| -------------------- | ------------------- | ------------------------- |
| 💰 Finance           | CFA Charterholder   | ROI, NPV, Sharpe Ratio    |
| 📈 Business Strategy | McKinsey Consultant | LTV, CAC, TAM             |
| ⚖️ Policy & Ethics   | Policy Researcher   | Impact, Equity, Precedent |
| 💻 Technology        | Principal Engineer  | Latency, SLA, Tech Debt   |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│   Query Input → Run Debate → [Debate | Analytics | History] │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                         │
│              debate_engine.py + scoring.py                   │
│   Planner → Round Loop → Critic → Fact-Check → Judge        │
└────────┬──────────────────┬──────────────────┬──────────────┘
         │                  │                  │
         ▼                  ▼                  ▼
   ┌───────────┐     ┌────────────┐    ┌──────────────────┐
   │   AGENT   │     │    RAG     │    │  MEMORY SYSTEM   │
   │   LAYER   │     │   LAYER    │    │                  │
   │  8 agents │     │ FAISS +    │    │ Short: session   │
   │  w/ roles │     │ Embeddings │    │ Long:  history   │
   └─────┬─────┘     └─────┬──────┘    └────────┬─────────┘
         │                 │                    │
         └─────────────────┴────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │      GROQ LLM API      │
              │  llama-3.3-70b (main)  │
              │  llama-3.1-8b (fast)   │
              └────────────────────────┘
```

### Debate Flow

```
Round 1 ──► Optimist ──► Skeptic ──► Analyst ──► Expert
              └──────────────── Critic scores ─────────────┘
              └──────────────── Fact-Checker verifies ──────┘

Round 2 ──► All agents read Round 1 → counterargue
Round 3 ──► All agents refine → closing statements

Post-Debate ──► ScoringEngine aggregates → Judge delivers verdict
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- Free [Groq API Key](https://console.groq.com) (no credit card needed)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/rohityadav483/multi-agent-debate.git
cd multi-agent-debate
```

### 2. Create virtual environment

```bash
# Windows
python -m venv env
env\Scripts\activate

# macOS / Linux
python -m venv env
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the app

```bash
streamlit run app.py

#To avoid warnings
streamlit run app.py --server.fileWatcherType none
```

Open **http://localhost:8501** in your browser. 🎉

---

## 🚀 Live Demo

> 🔗 **[multi-agent-debate.streamlit.app](https://multi-agent-debate.streamlit.app)**

**Try these example queries:**

- _"Should we invest in cryptocurrency in 2025?"_
- _"Should our startup pivot from B2C to B2B?"_
- _"Should AI-generated content require mandatory disclosure?"_
- _"Should we migrate our monolith to microservices?"_

---

## 📁 Project Structure

```
multi-agent-debate/
│
├── app.py                          # Streamlit entry point
├── requirements.txt
├── .env.example                    # API key template
│
├── backend/
│   ├── config.py                   # All constants & domain configs
│   ├── debate_engine.py            # Core orchestrator
│   ├── round_manager.py            # Per-round debate flow
│   └── scoring.py                  # Weighted score aggregation
│
├── agents/
│   ├── base_agent.py               # Abstract base class
│   ├── planner.py                  # Debate agenda generator
│   ├── optimist.py                 # Best-case advocate
│   ├── skeptic.py                  # Risk challenger
│   ├── analyst.py                  # Data-driven evidence
│   ├── domain_expert.py            # Domain specialist
│   ├── critic.py                   # Argument scorer (JSON)
│   ├── fact_checker.py             # Claim verifier (JSON)
│   └── judge.py                    # Final verdict (JSON)
│
├── rag/
│   ├── loader.py                   # PDF + TXT chunker
│   ├── embedder.py                 # SentenceTransformers
│   ├── vector_store.py             # FAISS operations
│   └── retriever.py                # Full RAG pipeline
│
├── memory/
│   ├── memory_schema.py            # Data models
│   ├── short_term.py               # Session state
│   └── long_term.py                # Persistent history
│
├── frontend/
│   ├── ui_components.py            # Shared widgets
│   ├── debate_view.py              # Transcript tab
│   └── analytics_view.py           # Charts tab
│
├── utils/
│   ├── llm_client.py               # Groq API wrapper
│   ├── prompt_builder.py           # Dynamic prompt builder
│   ├── domain_adapter.py           # Domain augmentation
│   └── logger.py                   # Centralised logging
│
└── data/
    ├── vector_index/               # FAISS index files
    ├── debate_history/             # JSON debate records
    └── uploads/                    # Temp PDF uploads
```

---

## 🛠️ Tech Stack

| Layer           | Technology           | Purpose                 |
| --------------- | -------------------- | ----------------------- |
| **LLM**         | Groq + LLaMA 3.3 70B | Debate agent inference  |
| **Fast LLM**    | LLaMA 3.1 8B Instant | Scoring & fact-checking |
| **Embeddings**  | SentenceTransformers | Local free embeddings   |
| **Vector DB**   | FAISS-CPU            | Similarity search       |
| **Frontend**    | Streamlit            | Web UI + charts         |
| **PDF Parsing** | PyPDF                | Document ingestion      |
| **Data**        | Pandas + NumPy       | Analytics               |
| **Config**      | python-dotenv        | Key management          |
| **Deployment**  | Streamlit Cloud      | Free hosting            |

---

## 🎓 Agent Personalities

| Agent           | Persona           | Temperature | Style                           |
| --------------- | ----------------- | ----------- | ------------------------------- |
| 🗂️ Planner      | Neutral organiser | 0.2         | Structured, methodical          |
| 🌟 Optimist     | Bold advocate     | 0.8         | Persuasive, opportunity-focused |
| 🔍 Skeptic      | Risk analyst      | 0.6         | Challenging, precise            |
| 📊 Analyst      | Data scientist    | 0.4         | Quantitative, evidence-first    |
| 🎓 Expert       | Domain specialist | 0.7         | Authoritative, technical        |
| ⚡ Critic       | Logic evaluator   | 0.3         | Impartial, structured           |
| ✅ Fact-Checker | Truth guardian    | 0.3         | Precise, audit-focused          |
| ⚖️ Judge        | Decision maker    | 0.2         | Decisive, balanced              |

---

## 📊 Scoring System

```
Per Agent Per Round:
  Logical Consistency  ──►  0-25 pts
  Evidence Usage       ──►  0-25 pts
  Relevance            ──►  0-25 pts
  Persuasiveness       ──►  0-25 pts
                            ────────
  Round Total          ──►  0-100 pts

Round Weights:
  Round 1  ──►  25%
  Round 2  ──►  35%
  Round 3  ──►  40%   (closing arguments weighted most)

Final Score = Σ (round_score × round_weight)
Max Score   = 100 pts
```

## 📋 Environment Variables

| Variable       | Required | Description                                              |
| -------------- | -------- | -------------------------------------------------------- |
| `GROQ_API_KEY` | ✅ Yes   | Get free at [console.groq.com](https://console.groq.com) |

---

## 🔧 Configuration

Key settings in `backend/config.py`:

```python
NUM_ROUNDS      = 3                        # Debate rounds
PRIMARY_MODEL   = "llama-3.3-70b-versatile"  # Main agent model
FAST_MODEL      = "llama-3.1-8b-instant"     # Scoring model
TOP_K_RETRIEVAL = 5                        # RAG chunks per query
MAX_LONG_TERM_DEBATES = 100               # Memory cap
```

---

## 📈 Example Output

```
⚖️ JUDGE'S FINAL DECISION
══════════════════════════════════════

DECISION:
The company should proceed with a phased B2B pivot over 6 months,
starting with 2-3 enterprise pilot clients before full migration.

CONFIDENCE:    82/100  (High)
RISK LEVEL:    Medium
WINNING AGENT: ANALYST

KEY REASONS:
  1. B2B LTV is 3.2x higher than current B2C average
  2. Competitor analysis shows 18-month window before market saturates
  3. Existing team has enterprise sales capability (2 of 5 members)

DISSENTING VIEW:
Skeptic raised valid concerns about 6-9 month revenue gap during
transition that requires bridge financing or careful cash management.
```

---

## 🗺️ Roadmap

- [ ] 🌐 Web search integration (live data via SerpAPI)
- [ ] 🎙️ Voice input (Whisper API)
- [ ] 📄 Export debate as PDF report
- [ ] 👤 Human-in-the-loop agent mode
- [ ] 🔄 Debate comparison across domains
- [ ] ⭐ Human feedback rating system
- [ ] 🌍 Multi-language support
- [ ] 🔌 FastAPI backend for REST access

---
