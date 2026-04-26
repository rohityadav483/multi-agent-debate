# =============================================================================
# backend/config.py
# Central configuration for the Multi-Agent Debate AI System
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# LLM API Configuration
# -----------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

PRIMARY_MODEL = "llama-3.3-70b-versatile"   # debate agents + JUDGE
FAST_MODEL    = "llama-3.1-8b-instant"       # critic + fact_checker only

# Model parameters
DEFAULT_TEMPERATURE = 0.7       # Balanced creativity
CRITIC_TEMPERATURE  = 0.3       # Low — critic needs consistency
JUDGE_TEMPERATURE   = 0.2       # Very low — judge needs determinism
MAX_TOKENS_DEBATE   = 600       # Per agent per round
MAX_TOKENS_JUDGE    = 800       # Judge needs more for summary
MAX_TOKENS_SCORE    = 1200       # Critic scoring

# -----------------------------------------------------------------------------
# Debate Engine Configuration
# -----------------------------------------------------------------------------

NUM_ROUNDS = 3                  # Number of debate rounds

DEBATE_AGENTS = [
    "optimist",
    "skeptic",
    "analyst",
    "domain_expert",
]

# Order in which agents speak each round
AGENT_SPEAKING_ORDER = ["optimist", "skeptic", "analyst", "domain_expert"]

# Agents that run AFTER the main debate agents each round
POST_ROUND_AGENTS = ["fact_checker", "critic"]

# -----------------------------------------------------------------------------
# Scoring Configuration
# -----------------------------------------------------------------------------

SCORING_METRICS = {
    "logical_consistency": 25,   # Max points
    "evidence_usage":      25,
    "relevance":           25,
    "persuasiveness":      25,
}
MAX_SCORE_PER_ROUND = 100
TOTAL_MAX_SCORE     = MAX_SCORE_PER_ROUND * NUM_ROUNDS   # 300

# Weights for aggregating scores across rounds (later rounds matter more)
ROUND_WEIGHTS = {
    1: 0.25,
    2: 0.35,
    3: 0.40,
}

# -----------------------------------------------------------------------------
# RAG Configuration
# -----------------------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # SentenceTransformers (free, local)
CHUNK_SIZE      = 500                    # Characters per chunk
CHUNK_OVERLAP   = 50                     # Overlap to preserve context
TOP_K_RETRIEVAL = 5                      # Number of chunks to retrieve

FAISS_INDEX_PATH    = "data/vector_index/debate.index"
FAISS_METADATA_PATH = "data/vector_index/metadata.json"

# -----------------------------------------------------------------------------
# Memory Configuration
# -----------------------------------------------------------------------------

DEBATE_HISTORY_PATH = "data/debate_history/"
MAX_LONG_TERM_DEBATES = 100              # Max stored debates before rotation
MEMORY_SIMILARITY_THRESHOLD = 0.75      # Cosine similarity for recall

# -----------------------------------------------------------------------------
# Domain Configuration
# -----------------------------------------------------------------------------
# Each domain gets a unique system context injected into agent prompts.
# This makes the same agent behave like a finance expert vs. a policy analyst.

DOMAINS = {
    "finance": {
        "label": "Finance & Investment",
        "context": (
            "You are operating in the financial domain. "
            "Focus on ROI, risk-adjusted returns, market trends, "
            "portfolio impact, regulatory compliance, and capital allocation. "
            "Use financial metrics (P/E ratio, CAGR, Sharpe ratio, etc.) where relevant."
        ),
        "example_topics": [
            "Should we invest in renewable energy stocks in 2025?",
            "Is cryptocurrency a viable long-term asset class?",
            "Should the company pursue an IPO this quarter?",
        ],
    },
    "business_strategy": {
        "label": "Business Strategy",
        "context": (
            "You are operating in the business strategy domain. "
            "Focus on competitive advantage, market positioning, SWOT analysis, "
            "operational efficiency, stakeholder value, and long-term growth. "
            "Reference frameworks like Porter's Five Forces, BCG matrix, etc."
        ),
        "example_topics": [
            "Should our startup pivot from B2C to B2B?",
            "Is acquiring a competitor better than organic growth?",
            "Should we expand internationally in the next 12 months?",
        ],
    },
    "policy_ethics": {
        "label": "Policy & Ethics",
        "context": (
            "You are operating in the policy and ethics domain. "
            "Focus on societal impact, fairness, legal frameworks, stakeholder rights, "
            "unintended consequences, and ethical principles (utilitarian, deontological). "
            "Balance short-term practicality with long-term societal good."
        ),
        "example_topics": [
            "Should AI-generated content require mandatory disclosure?",
            "Is universal basic income a viable policy?",
            "Should social media platforms be regulated as utilities?",
        ],
    },
    "technology": {
        "label": "Technology & Innovation",
        "context": (
            "You are operating in the technology and innovation domain. "
            "Focus on technical feasibility, scalability, security, adoption curves, "
            "disruption potential, and engineering trade-offs. "
            "Reference relevant tech stacks, architectures, and industry benchmarks."
        ),
        "example_topics": [
            "Should our platform migrate from monolith to microservices?",
            "Is quantum computing investment justified for our industry now?",
            "Should we build in-house AI or use third-party APIs?",
        ],
    },
}

DEFAULT_DOMAIN = "business_strategy"

# -----------------------------------------------------------------------------
# Fact-Checker Configuration
# -----------------------------------------------------------------------------

# Confidence thresholds for claim verification
FACT_CHECK_THRESHOLDS = {
    "verified":     0.80,   # Claim strongly supported by retrieved context
    "uncertain":    0.50,   # Partial support
    "unverified":   0.00,   # No support found — flagged as potential hallucination
}

# -----------------------------------------------------------------------------
# Judge Configuration
# -----------------------------------------------------------------------------

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]

CONFIDENCE_BANDS = {
    "high":   (75, 100),
    "medium": (50, 74),
    "low":    (0,  49),
}

# -----------------------------------------------------------------------------
# Application Configuration
# -----------------------------------------------------------------------------

APP_TITLE       = "Multi-Agent Debate AI"
APP_SUBTITLE    = "Intelligent Decision Making through Structured Agent Debate"
APP_VERSION     = "1.0.0"

LOG_LEVEL       = "INFO"
LOG_FILE        = "data/debate.log"

# Upload config
ALLOWED_FILE_TYPES  = ["pdf", "txt"]
MAX_UPLOAD_SIZE_MB  = 10

# UI timing
STREAM_DELAY_MS = 20    # Milliseconds between streamed characters in UI