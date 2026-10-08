from __future__ import annotations

import json
import logging
import re
from typing import Any

from django.conf import settings
from apps.model_router.router import ModelRouter

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_ANALYST = """You are Agent Query Analyst, a premier AI research methodologist at PrismSense.
Your mission is to collaborate with researchers, policy analysts, and think tank scholars.
You take unstructured, irregular, rambling, or embryonic ideas and transform them into precise, creative, and methodologically sound research inquiries.

You work in three interactive phases:
1. Question Formulation: Deconstruct raw narratives into distinct, rigorous, and novel questions across different complexity levels and audience perspectives.
2. Conceptual & Structural Deep-Dive: Analyze underlying theories, keywords/variables, and contextual problem mapping.
3. Methodological Guidance: Guide the user through both fundamental and advanced/complex research methodologies (econometric, qualitative, and mixed-methods).

Always respond in valid JSON format as requested by the user prompt.
Maintain an encouraging, intellectually rigorous, and structured tone.
"""


class QueryAnalystAgent:
    """Agent Query Analyst: Takes raw, irregular user ideas and structures them into
    creative research questions, audience perspectives, theoretical foundations,
    keyword mappings, and methodological blueprints.
    """

    def __init__(self):
        self.router = ModelRouter()

    def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        stage = payload.get("stage", "generate_questions")
        if stage == "generate_questions":
            idea = str(payload.get("idea") or payload.get("message") or "").strip()
            audience = str(payload.get("audience") or "all").strip().lower()
            return self.generate_questions(idea, audience)
        elif stage == "explore_topic":
            questions = payload.get("selected_questions") or []
            if isinstance(questions, str):
                questions = [questions]
            action = str(payload.get("action") or "theories").strip().lower()
            context = str(payload.get("context") or "").strip()
            return self.explore_topic(questions, action, context)
        else:
            # conversational chat
            history = payload.get("history") or []
            message = str(payload.get("message") or "").strip()
            selected_questions = payload.get("selected_questions") or []
            return self.chat(history, message, selected_questions)

    def generate_questions(self, idea: str, audience: str = "all") -> dict[str, Any]:
        """Convert a raw user idea into 4-6 diverse, creative, and complex research questions."""
        if not idea:
            return {
                "ok": False,
                "error": "Please provide an initial idea or narrative to analyze.",
            }

        # Try calling OpenRouter with the configured apodex model
        llm_result = self._call_llm_generate(idea, audience)
        if llm_result:
            return {
                "ok": True,
                "stage": "questions_ready",
                "summary": llm_result.get("summary", ""),
                "audience": audience,
                "questions": llm_result.get("questions", []),
                "chat_message": llm_result.get("chat_message", ""),
                "model_used": self.router.model if self.router.mode == "openrouter" else "deterministic-analyst",
            }

        # Fallback to deterministic research analyst generator
        fallback = self._fallback_generate_questions(idea, audience)
        return {
            "ok": True,
            "stage": "questions_ready",
            "summary": fallback["summary"],
            "audience": audience,
            "questions": fallback["questions"],
            "chat_message": fallback["chat_message"],
            "model_used": "deterministic-analyst",
        }

    def explore_topic(self, questions: list[str], action: str, context: str = "") -> dict[str, Any]:
        """Explore theories, keywords, problem mapping, or research methodology for selected questions."""
        if not questions:
            return {"ok": False, "error": "No questions were selected for deep-dive analysis."}

        action = action if action in ("theories", "keywords", "mapping", "methods") else "theories"

        llm_result = self._call_llm_explore(questions, action, context)
        if llm_result:
            return {
                "ok": True,
                "stage": "exploration_ready",
                "action": action,
                "questions": questions,
                "title": llm_result.get("title", f"Analysis: {action.title()}"),
                "content": llm_result.get("content", ""),
                "sections": llm_result.get("sections", []),
                "next_options": llm_result.get("next_options", self._default_next_options(action)),
                "chat_message": llm_result.get("chat_message", ""),
            }

        fallback = self._fallback_explore(questions, action)
        return {
            "ok": True,
            "stage": "exploration_ready",
            "action": action,
            "questions": questions,
            "title": fallback["title"],
            "content": fallback["content"],
            "sections": fallback["sections"],
            "next_options": fallback["next_options"],
            "chat_message": fallback["chat_message"],
        }

    def chat(self, history: list[dict], message: str, selected_questions: list[str]) -> dict[str, Any]:
        """Handle conversational follow-up questions from the user."""
        if not message:
            return {"ok": False, "error": "Message cannot be empty."}

        llm_reply = self._call_llm_chat(history, message, selected_questions)
        if llm_reply:
            return {
                "ok": True,
                "reply": llm_reply,
                "chat_message": llm_reply,
            }

        # Fallback conversational response
        fallback_reply = self._fallback_chat(message, selected_questions)
        return {
            "ok": True,
            "reply": fallback_reply,
            "chat_message": fallback_reply,
        }

    # ---------------- OpenRouter LLM Callers ----------------

    def _call_llm_generate(self, idea: str, audience: str) -> dict[str, Any] | None:
        if self.router.mode != "openrouter" or not settings.OPENROUTER_API_KEY:
            return None

        audience_desc = {
            "all": "Multidisciplinary academic, policy, and practitioner audience",
            "policy": "Policymakers, government officials, and institutional regulators looking for actionable levers",
            "academic": "Scholars, journal editors, and theorists requiring rigorous causal identification and conceptual novelty",
            "ngo": "Development organizations, humanitarian practitioners, and field implementation teams",
            "industry": "Corporate strategy, private R&D, and market investment analysts",
            "public": "Investigative journalists, civil society advocates, and the broader informed public",
        }.get(audience, audience)

        prompt = f"""{SYSTEM_PROMPT_ANALYST}

TASK: The user has given you a raw, potentially unstructured idea or thought.
Transform this idea into 4 to 6 creative, provocative, and methodologically sound research questions tailored for the specified audience.

Raw User Narrative:
\"\"\"{idea}\"\"\"

Target Audience: {audience_desc}

OUTPUT REQUIREMENTS:
Return strict JSON with this exact schema:
{{
  "summary": "1-2 sentences summarizing the core tension or intuition behind the user's idea.",
  "questions": [
    {{
      "id": "q1",
      "question": "Fully articulated research question string",
      "angle": "e.g. Causal & Econometric / Institutional Tension / Distributional & Equity / Counter-Intuitive Friction / Policy & Implementation",
      "complexity": "Fundamental" | "Intermediate" | "Complex & Novel",
      "why_it_matters": "1 concise sentence explaining why this specific inquiry produces high-impact knowledge."
    }}
  ],
  "chat_message": "Warm, intellectual greeting explaining how you structured their thought into these distinct angles and prompting them to select one or multiple questions to proceed."
}}
JSON ONLY:
"""
        reply = self.router._post(prompt, temperature=0.3, timeout=30, long_form=True)
        if reply:
            data = self.router._clean_json(reply)
            if data and isinstance(data.get("questions"), list) and len(data["questions"]) > 0:
                return data
        return None

    def _call_llm_explore(self, questions: list[str], action: str, context: str) -> dict[str, Any] | None:
        if self.router.mode != "openrouter" or not settings.OPENROUTER_API_KEY:
            return None

        q_list_text = "\n".join(f"- {q}" for q in questions)
        action_instructions = {
            "theories": "Provide the foundational theoretical frameworks, competing paradigms, key academic literature, and conceptual models explaining this inquiry.",
            "keywords": "Provide a complete keyword & semantic analysis: independent/dependent variables, search strings for database retrieval, synonyms, and operational indicators.",
            "mapping": "Provide a contextual problem mapping: root causes, stakeholder landscape, structural bottlenecks, institutional incentives, and systemic friction points.",
            "methods": "Provide a comprehensive methodological blueprint covering both Fundamental Methods (surveys, secondary data regression, descriptive stats) and Complex/Advanced Methods (DiD, IV, synthetic control, mixed-methods triangulation, causal DAGs), with step-by-step guidance on how to apply them to these questions.",
        }.get(action, "Provide an in-depth analysis of this topic.")

        prompt = f"""{SYSTEM_PROMPT_ANALYST}

TASK: Deep-dive into the '{action}' dimension for the selected research question(s).
Selected Research Question(s):
{q_list_text}

Additional Context:
{context or 'None provided.'}

Specific Directive for '{action}':
{action_instructions}

OUTPUT REQUIREMENTS:
Return strict JSON:
{{
  "title": "A compelling title for this deep-dive section",
  "content": "Rich, multi-paragraph markdown analysis detailing the requested {action}. Use bolding, bullet points, and clear explanations.",
  "sections": [
    {{
      "heading": "Subheading",
      "body": "Detailed analytical breakdown",
      "tags": ["Tag1", "Tag2"]
    }}
  ],
  "next_options": [
    {{"id": "theories", "label": "Explain theoretical frameworks", "icon": "book"}},
    {{"id": "keywords", "label": "Keyword & variable analysis", "icon": "search"}},
    {{"id": "mapping", "label": "Contextual problem mapping", "icon": "map"}},
    {{"id": "methods", "label": "Methodological blueprint", "icon": "tool"}}
  ],
  "chat_message": "Conversational summary explaining what was analyzed and inviting the user to explore methods or launch the query into Research Desk or Orchestrator."
}}
JSON ONLY:
"""
        reply = self.router._post(prompt, temperature=0.3, timeout=35, long_form=True)
        if reply:
            data = self.router._clean_json(reply)
            if data and data.get("content"):
                return data
        return None

    def _call_llm_chat(self, history: list[dict], message: str, selected_questions: list[str]) -> str | None:
        if self.router.mode != "openrouter" or not settings.OPENROUTER_API_KEY:
            return None

        context_q = "\n".join(f"- {q}" for q in selected_questions) if selected_questions else "None selected yet."
        history_formatted = "\n".join(
            f"{'User' if h.get('role') == 'user' else 'Agent'}: {h.get('content')}"
            for h in history[-6:]
        )

        prompt = f"""{SYSTEM_PROMPT_ANALYST}

Active Selected Questions:
{context_q}

Recent Conversation:
{history_formatted}

User Message:
\"{message}\"

Respond helpfully as Agent Query Analyst. Provide clear, rigorous, and insightful guidance. Use polished Markdown. Keep response under 350 words unless the user explicitly requested an exhaustive breakdown.
"""
        return self.router._post(prompt, temperature=0.3, timeout=25, long_form=False)

    # ---------------- Deterministic Fallback Generators ----------------

    def _fallback_generate_questions(self, idea: str, audience: str) -> dict[str, Any]:
        """Generate high-quality questions deterministically when OpenRouter is unavailable."""
        clean_idea = re.sub(r"\s+", " ", idea).strip()
        words = [w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", clean_idea)]
        
        # Extract potential key entities or themes
        topic_phrase = clean_idea[:90] + ("..." if len(clean_idea) > 90 else "")

        questions = [
            {
                "id": "q1",
                "question": f"To what extent does headline growth or progress in '{clean_idea[:45]}' translate into equitable structural outcomes for marginalized populations?",
                "angle": "Structural & Distributional Equity",
                "complexity": "Intermediate",
                "why_it_matters": "Isolates aggregate metrics from real distributional welfare gains, revealing hidden inequalities."
            },
            {
                "id": "q2",
                "question": f"What are the direct causal transmission mechanisms linking institutional interventions to measurable outcomes in this context?",
                "angle": "Causal & Econometric Identification",
                "complexity": "Complex & Novel",
                "why_it_matters": "Establishes causal attribution while filtering out unobserved confounding factors and macroeconomic shocks."
            },
            {
                "id": "q3",
                "question": f"How do informal institutional frictions and local political economy incentives distort the intended impact of policies in this domain?",
                "angle": "Political Economy & Institutional Friction",
                "complexity": "Complex & Novel",
                "why_it_matters": "Explains why technically sound policy interventions frequently produce unintended sub-optimal outcomes."
            },
            {
                "id": "q4",
                "question": f"What key empirical indicators and longitudinal trends best quantify the baseline vs. target shifts over the last decade?",
                "angle": "Empirical Baseline & Indicator Trends",
                "complexity": "Fundamental",
                "why_it_matters": "Grounds theoretical debate in verifiable numeric time-series and empirical observation."
            },
            {
                "id": "q5",
                "question": f"Under what boundary conditions and external shocks does the prevailing development model in this area risk systemic fragility?",
                "angle": "Counter-Narrative & Stress Testing",
                "complexity": "Complex & Novel",
                "why_it_matters": "Tests the resilience of conventional assumptions against emerging macroeconomic and ecological volatility."
            }
        ]

        summary = f"Your narrative touches on the core relationship between intervention mechanisms, institutional friction, and long-term distributional impact in {topic_phrase}."
        chat_msg = (
            f"I analyzed your idea and formulated **{len(questions)} distinct research questions**, spanning empirical baseline tracking, causal econometrics, and structural political economy. "
            f"You can choose **one question** for focused inquiry, or select **multiple questions** to compare interconnected dimensions. Let me know which direction resonates with you!"
        )

        return {
            "summary": summary,
            "questions": questions,
            "chat_message": chat_msg,
        }

    def _fallback_explore(self, questions: list[str], action: str) -> dict[str, Any]:
        primary_q = questions[0]
        
        if action == "theories":
            return {
                "title": "Theoretical Frameworks & Academic Foundations",
                "content": f"""### Grounding Theoretical Paradigms

When analyzing your question: *"{primary_q}"*, several complementary and competing theoretical schools provide essential analytical lenses:

1. **Sen's Capability Approach & Multidimensional Welfare Theory**
   - *Core Premise*: Welfare is not solely command over commodities or headline monetary growth, but the substantive freedom of people to lead lives they value.
   - *Application*: Enables distinguishing between monetary growth and systemic capability enhancement.

2. **New Institutional Economics (North, Acemoglu & Robinson)**
   - *Core Premise*: Institutions (both formal regulations and informal social norms) dictate incentive structures and transaction costs.
   - *Application*: Unpacks why policy announcements fail when underlying enforcement mechanisms and rent-seeking incentives remain misaligned.

3. **Information Asymmetry & Credit Market Rationing (Stiglitz & Weiss)**
   - *Core Premise*: Imperfect information produces adverse selection and moral hazard, leading to structural market failures.
   - *Application*: Explains institutional exclusion, informal borrowing costs, and distributional divergence.
""",
                "sections": [
                    {
                        "heading": "Capability vs. Utility Paradigm",
                        "body": "Frames the inquiry beyond GDP growth toward localized resilience and human agency.",
                        "tags": ["Amartya Sen", "Multidimensional Poverty", "Welfare Economics"]
                    },
                    {
                        "heading": "Institutional Enforcement & Rent-Seeking",
                        "body": "Identifies political economy bottlenecks that distort policy implementation.",
                        "tags": ["Institutions", "Governance", "Political Economy"]
                    }
                ],
                "next_options": self._default_next_options(action),
                "chat_message": "I mapped out the core theoretical foundations relevant to your selected questions. Next, would you like to explore the **methodological blueprint** (fundamental vs. complex methods) or conduct a **keyword & semantic analysis**?"
            }

        elif action == "keywords":
            return {
                "title": "Keyword & Semantic Variable Mapping",
                "content": f"""### Variables and Retrieval Lexicon

For rigorous evidence retrieval and empirical modeling of *"{primary_q}"*, here is the structured variable decomposition:

#### 1. Core Variables
* **Dependent Variables (Outcomes)**: Inclusive growth rate, Gini coefficient of income/consumption, multidimensional poverty index (MPI), household asset accumulation, labor force participation.
* **Independent / Explanatory Variables**: Infrastructure capital expenditure, targeted social safety net transfers, financial inclusion density, microcredit penetration, trade openness.
* **Confounding & Control Variables**: External terms-of-trade shocks, inflation (CPI), regional vulnerability, demographic dependency ratio.

#### 2. Search Syntax for Evidence Retrieval
* **Boolean String (Global Repositories)**:
  `("inclusive growth" OR "inequality" OR "poverty alleviation") AND ("institutional capacity" OR "development policy") AND ("Bangladesh" OR "South Asia")`
* **Numeric Indicators**: World Bank WDI series `SI.POV.NAHC`, `NY.GDP.MKTP.KD.ZG`, BBS HIES consumption quintiles.
""",
                "sections": [
                    {
                        "heading": "Operational Dependent Variables",
                        "body": "Key indicators to measure actual impact on welfare and structural change.",
                        "tags": ["HIES", "MPI", "Gini Index", "Labor Share"]
                    },
                    {
                        "heading": "Search Strings & Synonyms",
                        "body": "Optimized queries for extracting verified literature from the Evidence Library.",
                        "tags": ["Boolean Retrieval", "Evidence Library", "Bibliometrics"]
                    }
                ],
                "next_options": self._default_next_options(action),
                "chat_message": "Here is the keyword and variable breakdown. You can now examine the **methodological approaches** or map the **contextual problem landscape**."
            }

        elif action == "mapping":
            return {
                "title": "Contextual Problem & Ecosystem Mapping",
                "content": f"""### Problem Tree & Contextual Ecosystem

Mapping the contextual landscape behind *"{primary_q}"*:

#### Root Causes & Structural Drivers
* **Structural Labor Market Dualism**: Disproportionate concentration of workforce in low-productivity informal occupations without social protection.
* **Fiscal Space Constraints**: Low tax-to-GDP ratio limiting public investments in quality education, universal healthcare, and climate adaptation.
* **Spatial Disparity**: Pronounced divergence between metropolitan economic hubs and vulnerable agrarian/coastal zones.

#### Stakeholder Incentives & Political Dynamics
* **Government Regulators**: Prioritize short-term macroeconomic stability and revenue mobilization.
* **Development Partners & NGOs**: Emphasize targeted grant programs, yet often encounter scale and sustainability barriers.
* **Private Enterprise**: Constrained by logistics costs, regulatory compliance burdens, and access to formal credit.
""",
                "sections": [
                    {
                        "heading": "Systemic Friction Points",
                        "body": "Root causes that perpetuate divergence between macro indicators and micro welfare.",
                        "tags": ["Fiscal Capacity", "Informality", "Spatial Disparities"]
                    },
                    {
                        "heading": "Institutional Stakeholders",
                        "body": "Alignment and misalignment of incentives across government, NGOs, and civil society.",
                        "tags": ["Policy Incentives", "Governance", "Coordination"]
                    }
                ],
                "next_options": self._default_next_options(action),
                "chat_message": "The contextual landscape and problem tree are mapped. Let's look at what **fundamental and complex methods** we can apply to investigate this rigorously."
            }

        else: # methods
            return {
                "title": "Methodological Blueprint: Fundamental & Complex Approaches",
                "content": f"""### Methodological Framework

To investigate *"{primary_q}"* with methodological rigor, here is a progressive research strategy combining fundamental and advanced techniques:

---

### 1. Fundamental Methods (Accessible & Foundational)
* **Secondary Time-Series Trend Analysis**:
  * *Method*: Collate BBS HIES survey rounds (2010, 2016, 2022) with World Bank WDI national accounts.
  * *Application*: Calculate Compound Annual Growth Rate (CAGR), Growth Incidence Curves (GIC), and poverty elasticity of growth.
* **Cross-Sectional Comparative Benchmarking**:
  * *Method*: Compare regional/district indicators against national benchmarks across key welfare metrics.
  * *Application*: Highlights subnational variation and identifies lagging administrative clusters.
* **Structured Desk Synthesis & Bibliometric Review**:
  * *Method*: Systematic screening of peer-reviewed literature, policy evaluation reports, and multilateral assessments.
  * *Application*: Synthesizes established consensus and unresolved policy debates.

---

### 2. Complex & Advanced Methods (Cutting-Edge & Causal)
* **Difference-in-Differences (DiD) / Event Study Designs**:
  * *Method*: Exploit spatial or temporal rollouts of specific policy interventions (e.g. social safety net expansion or major infrastructure completion) to estimate counterfactual causal impact.
  * *Verification*: Parallel trends testing, falsification tests using pre-treatment periods.
* **Instrumental Variables (IV) / 2SLS Regression**:
  * *Method*: Address endogeneity and reverse causality in financial inclusion or policy adoption using geographical or historical instruments.
  * *Application*: Isolates true exogenous impact from selection bias.
* **Mixed-Methods Sequential Triangulation (Qual-Quant Integration)**:
  * *Method*: Pair econometric microdata with qualitative Key Informant Interviews (KII) and Focus Group Discussions (FGD).
  * *Application*: Explains *why* certain quantitative patterns exist and illuminates informal institutional dynamics invisible in administrative data.

---

### 3. Step-by-Step Action Protocol
1. **Formulate Bounded Hypothesis**: Set strict time boundaries (e.g., 2015–2024) and target population.
2. **Retrieve Baseline Evidence**: Pin relevant studies from the Evidence Library.
3. **Execute Synthesis**: Run either the **Research Desk** (for a focused brief) or the **Master Research Orchestrator** (for comprehensive multi-perspective analysis).
""",
                "sections": [
                    {
                        "heading": "Fundamental Approaches",
                        "body": "Descriptive statistics, growth incidence curves, and secondary database collation.",
                        "tags": ["Growth Incidence", "BBS HIES", "Time-Series"]
                    },
                    {
                        "heading": "Advanced Causal Inference",
                        "body": "Difference-in-Differences, Instrumental Variables, and Mixed-Methods Triangulation.",
                        "tags": ["Causal Inference", "DiD", "2SLS", "Triangulation"]
                    }
                ],
                "next_options": [
                    {"id": "theories", "label": "Review Theoretical Frameworks", "icon": "book"},
                    {"id": "keywords", "label": "Review Key Variables", "icon": "search"},
                    {"id": "mapping", "label": "Contextual Problem Mapping", "icon": "map"},
                ],
                "chat_message": "I've outlined both the fundamental and advanced methods suitable for your inquiry. You can now take this question directly into the **Research Desk** or **Master Orchestrator** to synthesize an evidence-grounded report!"
            }

    def _fallback_chat(self, message: str, selected_questions: list[str]) -> str:
        q_context = f"regarding '{selected_questions[0]}'" if selected_questions else "on your research narrative"
        return (
            f"Regarding your point: *\"{message}\"*, that is a crucial dimension {q_context}. "
            "In research methodology, exploring this angle requires clarifying the independent mechanism versus observable outcome. "
            "Would you like me to formulate an additional sub-question for this specific nuance, or should we examine how to test it methodologically?"
        )

    def _default_next_options(self, current_action: str) -> list[dict[str, str]]:
        all_opts = [
            {"id": "theories", "label": "Explain the theories more", "icon": "book"},
            {"id": "keywords", "label": "Keywords & variable analysis", "icon": "search"},
            {"id": "mapping", "label": "Map contextual problem landscape", "icon": "map"},
            {"id": "methods", "label": "Fundamental & complex methods", "icon": "tool"},
        ]
        return [opt for opt in all_opts if opt["id"] != current_action]
