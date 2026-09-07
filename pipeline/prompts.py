"""
pipeline/prompts.py

Prompt string constants used by the fact extraction pipeline.
No functions — only string templates to be formatted by callers.
"""

# ---------------------------------------------------------------------------
# FACT_EXTRACTION_PROMPT
# Format keys: {document_name}, {page_number}, {text}
# ---------------------------------------------------------------------------

FACT_EXTRACTION_PROMPT = """\
You are a precise financial and economic fact extraction engine.

Document: {document_name}
Page: {page_number}

Text excerpt:
\"\"\"
{text}
\"\"\"

Your task:
1. Read the text excerpt above carefully.
2. Extract EVERY meaningful, verifiable fact — both numerical and semantic.
   - Numerical facts include: revenue, profit, headcount, percentages, growth rates, dates, interest rates, market share figures, valuations, volume metrics.
   - Semantic facts include: who holds an executive position, what a company claims about its strategy or outlook, what policy is currently in effect, what ratings or certifications apply, what agreements or partnerships exist.
3. For EACH fact, return a JSON object with exactly these keys:
   - "fact_type": A short snake_case label you invent that describes the kind of fact (e.g. "revenue_metric", "executive_role", "gdp_growth_rate", "policy_rate", "market_share", "headcount", "net_profit", "ebitda_margin"). Be specific and consistent.
   - "subject": The entity the fact is about (e.g. "Delhivery", "Reserve Bank of India", "India GDP").
   - "predicate": What is being stated about the subject (e.g. "reported revenue for FY24", "appointed as CFO", "raised repo rate to").
   - "value": The exact value or claim as stated (e.g. "₹7,225 crore", "Amit Agarwal", "6.5%", "will expand into tier-3 cities").
   - "unit": The unit of measurement if applicable (e.g. "crore INR", "percent", "basis points"), otherwise null.
   - "period": The time period the fact refers to if mentioned (e.g. "FY2024", "Q4 FY24", "2024-25", "as of March 2024"), otherwise null.
   - "scope": Any explicit scope qualifier (e.g. "consolidated", "standalone", "national", "Q4 only", "YoY"), otherwise null.
   - "evidence_quote": The EXACT verbatim substring from the text above that contains this fact. This string MUST appear character-for-character in the provided text.
   - "confidence": A float between 0.0 and 1.0. Use lower values (< 0.7) for: ambiguous phrasing, table cells lacking surrounding context, forward-looking statements, or approximated figures. Use higher values (>= 0.85) for clearly stated, precise, dated facts.

Rules:
- Return ONLY a valid JSON array of fact objects. No prose, no markdown code fences, no explanation, no commentary.
- Do not invent or infer facts not explicitly present in the text.
- If the same fact appears in both a table and surrounding prose, extract it once.
- If no facts are found in this excerpt, return an empty JSON array: []
"""


# ---------------------------------------------------------------------------
# DOCUMENT_METADATA_PROMPT
# Format key: {text}
# ---------------------------------------------------------------------------

DOCUMENT_METADATA_PROMPT = """\
You are a document classification and metadata extraction engine.

Below is the opening text (first 1-2 pages) of a document:
\"\"\"
{text}
\"\"\"

Extract the following metadata and return it as a single valid JSON object with exactly these keys:
- "title": The full title of the document as it appears in the text (string).
- "organization": The organization, company, or institution that produced or is the primary subject of this document (string).
- "document_type": One of the following labels that best describes this document: "annual_report", "prospectus", "earnings_presentation", "economic_survey", "policy_report", "research_report", "other".
- "period_covered": The fiscal year, quarter, or date range this document covers (e.g. "FY2024", "Q4 FY2023-24", "2024-25"), or null if not determinable.
- "publication_date": The publication or release date if mentioned (ISO 8601 format preferred, e.g. "2024-07-15"), or null if not determinable.

Return ONLY valid JSON. No prose, no markdown fences, no explanation.
"""


# ---------------------------------------------------------------------------
# RELATIONSHIP_PROMPT
# Format keys: {fact_a_json}, {fact_b_json}
# ---------------------------------------------------------------------------

RELATIONSHIP_PROMPT = """\
You are a financial and economic fact relationship analyst.

You will be given two structured facts extracted from different source documents. \
Your task is to determine the relationship between them.

Fact A:
{fact_a_json}

Fact B:
{fact_b_json}

Determine the relationship between Fact A and Fact B. Choose exactly one of:

- "corroborates": Both facts make the same or fully consistent claim about the same subject. \
  They may use slightly different wording but agree on the underlying data point.

- "contradicts": The facts make genuinely conflicting claims about the same subject, \
  and the conflict CANNOT be explained by differences in time period, unit, scope, or \
  reporting standard. This is a true data conflict.

- "reconcilable": The facts appear to conflict numerically or semantically, but the \
  conflict CAN be explained by a clear difference in time period (e.g. FY23 vs FY24), \
  unit (e.g. crore vs million), scope (e.g. consolidated vs standalone), or reporting \
  standard. The explanation must cite the specific reconciling factor.

- "unrelated": The two facts are about completely different subjects or topics with no \
  meaningful connection worth surfacing to an analyst.

Rules:
- NEVER return "unrelated" if both facts share the same subject AND predicate — you must \
  reason through them and return corroborates, contradicts, or reconcilable.
- Your explanation must be 1-3 sentences in plain English. It must cite the specific \
  agreement or difference (e.g., mention the exact values, the unit difference, the \
  time period mismatch, or the scope qualifier).
- Base your reasoning only on the information present in the two fact objects above. \
  Do not hallucinate or assume context not provided.

Return ONLY a valid JSON object with exactly these keys:
- "relationship_type": one of "corroborates", "contradicts", "reconcilable", "unrelated"
- "confidence": float between 0.0 and 1.0 — your confidence in this classification
- "explanation": string, 1-3 sentences

No prose, no markdown fences, no extra keys.
"""
