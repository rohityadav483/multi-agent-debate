# =============================================================================
# utils/domain_adapter.py
# Runtime domain switching + per-domain prompt augmentation
# =============================================================================

from backend.config import DOMAINS, DEFAULT_DOMAIN
from utils.logger import get_logger

logger = get_logger(__name__)


class DomainAdapter:
    """
    Provides domain-specific prompt augmentations beyond the base
    context already in config.py.

    Used by PromptBuilder to inject:
        - Domain-specific vocabulary hints
        - Relevant frameworks / methodologies
        - Output format expectations per domain
        - Agent-specific domain instructions

    Usage:
        adapter = DomainAdapter("finance")
        extra   = adapter.get_agent_hint("skeptic")
        metrics = adapter.get_key_metrics()
    """

    # ------------------------------------------------------------------
    # Per-domain agent hints
    # Extra instructions injected into agent prompts on top of base persona
    # ------------------------------------------------------------------
    _AGENT_HINTS = {
        "finance": {
            "optimist":      "Cite specific financial metrics: IRR, NPV, CAGR, Sharpe ratio. Reference bull-case analyst forecasts.",
            "skeptic":       "Focus on downside risk, liquidity risk, regulatory exposure, and black-swan scenarios.",
            "analyst":       "Use actual market data, P/E ratios, revenue growth rates, sector benchmarks. Quantify everything.",
            "domain_expert": "Speak as a CFA charterholder. Reference CFA Institute standards, IFRS/GAAP implications, portfolio theory.",
            "judge":         "Frame decision in terms of risk-adjusted return. State risk level using standard financial risk taxonomy.",
        },
        "business_strategy": {
            "optimist":      "Reference Blue Ocean Strategy, first-mover advantage, network effects, and TAM/SAM/SOM.",
            "skeptic":       "Apply Porter's Five Forces. Challenge assumptions about market size, CAC, and competitive moats.",
            "analyst":       "Use unit economics: LTV, CAC, payback period, gross margin, burn rate, runway.",
            "domain_expert": "Speak as a McKinsey principal. Reference BCG matrix, MECE frameworks, and value chain analysis.",
            "judge":         "Frame decision in terms of strategic fit, execution risk, and time-to-value.",
        },
        "policy_ethics": {
            "optimist":      "Argue from utilitarian perspective — greatest good for greatest number. Cite positive precedents.",
            "skeptic":       "Apply deontological critique. Identify who bears the burden and who benefits asymmetrically.",
            "analyst":       "Cite policy research, academic studies, and empirical evidence from similar implementations.",
            "domain_expert": "Speak as a senior policy researcher. Reference constitutional law, precedent, and implementation challenges.",
            "judge":         "Balance competing rights. Explicitly acknowledge trade-offs between liberty, equity, and efficiency.",
        },
        "technology": {
            "optimist":      "Argue from scalability and innovation angle. Reference successful adoption curves and ROI from similar migrations.",
            "skeptic":       "Focus on technical debt, migration risk, team capability gaps, and vendor lock-in.",
            "analyst":       "Use engineering metrics: latency, throughput, uptime SLAs, cost per request, dev velocity.",
            "domain_expert": "Speak as a principal engineer or CTO. Reference system design principles, CAP theorem, and industry benchmarks.",
            "judge":         "Frame decision around build-vs-buy trade-offs, total cost of ownership, and engineering team capacity.",
        },
    }

    # ------------------------------------------------------------------
    # Key metrics per domain — injected into Analyst prompts
    # ------------------------------------------------------------------
    _KEY_METRICS = {
        "finance": [
            "Return on Investment (ROI)",
            "Net Present Value (NPV)",
            "Internal Rate of Return (IRR)",
            "Sharpe Ratio / Risk-adjusted return",
            "Price-to-Earnings (P/E) ratio",
            "Compound Annual Growth Rate (CAGR)",
            "Debt-to-Equity ratio",
            "Free Cash Flow (FCF)",
        ],
        "business_strategy": [
            "Customer Acquisition Cost (CAC)",
            "Lifetime Value (LTV) / LTV:CAC ratio",
            "Total Addressable Market (TAM)",
            "Gross Margin %",
            "Monthly Recurring Revenue (MRR)",
            "Churn Rate",
            "Payback Period",
            "Market Share %",
        ],
        "policy_ethics": [
            "Affected population size",
            "Cost-benefit ratio",
            "Compliance / enforcement rate",
            "Equity impact (Gini coefficient)",
            "Implementation timeline",
            "Precedent alignment",
            "Stakeholder approval rate",
            "Measurable outcome indicators (KPIs)",
        ],
        "technology": [
            "System latency (p50 / p99)",
            "Throughput (requests/sec)",
            "Uptime SLA (%)",
            "Cost per request",
            "Developer velocity (deploys/day)",
            "Mean Time to Recovery (MTTR)",
            "Technical debt ratio",
            "Test coverage %",
        ],
    }

    # ------------------------------------------------------------------
    # Frameworks per domain — referenced in Expert and Critic prompts
    # ------------------------------------------------------------------
    _FRAMEWORKS = {
        "finance": [
            "Modern Portfolio Theory (Markowitz)",
            "Capital Asset Pricing Model (CAPM)",
            "Discounted Cash Flow (DCF) analysis",
            "Black-Scholes (options pricing)",
            "Efficient Market Hypothesis",
        ],
        "business_strategy": [
            "Porter's Five Forces",
            "BCG Growth-Share Matrix",
            "SWOT / PESTLE Analysis",
            "Blue Ocean Strategy",
            "Jobs-to-be-Done (JTBD) Framework",
            "Ansoff Matrix",
        ],
        "policy_ethics": [
            "Utilitarian / Consequentialist ethics",
            "Kantian / Deontological ethics",
            "Rawlsian Justice (Veil of Ignorance)",
            "Cost-Benefit Analysis (CBA)",
            "Regulatory Impact Assessment (RIA)",
        ],
        "technology": [
            "CAP Theorem",
            "12-Factor App methodology",
            "Domain-Driven Design (DDD)",
            "TOGAF Enterprise Architecture",
            "Site Reliability Engineering (SRE)",
        ],
    }

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, domain: str = DEFAULT_DOMAIN):
        if domain not in DOMAINS:
            logger.warning(f"DomainAdapter: unknown domain '{domain}' — using '{DEFAULT_DOMAIN}'")
            domain = DEFAULT_DOMAIN
        self.domain = domain
        logger.info(f"DomainAdapter initialised | domain={domain}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_agent_hint(self, agent_name: str) -> str:
        """
        Returns a domain-specific extra instruction for a given agent.
        Injected by PromptBuilder into the agent's system prompt.

        Example:
            adapter.get_agent_hint("skeptic")
            → "Focus on downside risk, liquidity risk, regulatory exposure..."
        """
        return self._AGENT_HINTS.get(self.domain, {}).get(agent_name, "")

    def get_key_metrics(self) -> list[str]:
        """
        Returns the list of key metrics relevant to the current domain.
        Injected into the Analyst agent's prompt.
        """
        return self._KEY_METRICS.get(self.domain, [])

    def get_frameworks(self) -> list[str]:
        """
        Returns relevant analytical frameworks for the current domain.
        Injected into Domain Expert and Critic prompts.
        """
        return self._FRAMEWORKS.get(self.domain, [])

    def get_metrics_block(self) -> str:
        """
        Formats key metrics as a prompt-ready string block.

        Example output:
            'Key metrics to reference: ROI, NPV, IRR, Sharpe Ratio, ...'
        """
        metrics = self.get_key_metrics()
        if not metrics:
            return ""
        return "Key metrics to reference: " + ", ".join(metrics) + "."

    def get_frameworks_block(self) -> str:
        """
        Formats frameworks as a prompt-ready string block.
        """
        frameworks = self.get_frameworks()
        if not frameworks:
            return ""
        return "Relevant frameworks: " + ", ".join(frameworks) + "."

    def build_augmentation(self, agent_name: str) -> str:
        """
        Builds the full domain augmentation string for a given agent.
        Combines: agent hint + metrics block + frameworks block.

        Used by PromptBuilder._build_system() to append after domain context.

        Returns:
            Multi-line string ready for system prompt injection.
            Empty string if no augmentations apply.
        """
        parts = []

        hint = self.get_agent_hint(agent_name)
        if hint:
            parts.append(hint)

        # Metrics only for analyst
        if agent_name == "analyst":
            metrics = self.get_metrics_block()
            if metrics:
                parts.append(metrics)

        # Frameworks for expert and critic
        if agent_name in ("domain_expert", "critic"):
            fw = self.get_frameworks_block()
            if fw:
                parts.append(fw)

        return "\n".join(parts)

    def switch_domain(self, new_domain: str):
        """Hot-swap domain without recreating the adapter."""
        if new_domain not in DOMAINS:
            logger.warning(f"DomainAdapter.switch_domain: '{new_domain}' unknown — ignored.")
            return
        logger.info(f"DomainAdapter: switched {self.domain} → {new_domain}")
        self.domain = new_domain

    @staticmethod
    def list_domains() -> list[str]:
        """Returns all valid domain keys."""
        return list(DOMAINS.keys())

    def __repr__(self):
        return f"DomainAdapter(domain={self.domain})"