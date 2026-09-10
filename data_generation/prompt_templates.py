"""Report + news prompt templates as constants."""

REPORT_PROMPT_TEMPLATE = """You are generating SYNTHETIC and entirely FICTIONAL quarterly earnings data for an academic demonstration project. Do not reference or copy real company financials.

Generate a quarterly financial report for:
- Company Name: {company_name}
- Sector: {sector}
- Ticker: {ticker}
- Quarter: {quarter}

Important instructions:
- This is SYNTHETIC/fictional data for an academic demo only.
- Performance levels should vary realistically across companies (some strong, some average, some struggling).
- Use plausible but invented figures appropriate for the sector.
- Return ONLY valid JSON with no markdown, code fences, or extra commentary.

The JSON object must contain exactly these keys:
- company_name (string)
- sector (string)
- ticker (string)
- quarter (string)
- revenue (string, e.g. "$1.2B")
- revenue_growth_yoy (string, e.g. "12.5%")
- net_profit (string)
- profit_margin (string)
- key_highlights (array of strings, 3-5 items)
- management_commentary (string, 2-3 sentences)
- risks_and_challenges (array of strings, 2-4 items)
- outlook_next_quarter (string, 2-3 sentences)
"""

NEWS_PROMPT_TEMPLATE = """You are generating SYNTHETIC and entirely FICTIONAL news snippets for an academic demonstration project. Do not reference real events or real companies.

Generate exactly 3 news snippets about {company_name} in the {sector} sector during {quarter}.

Important instructions:
- This is SYNTHETIC/fictional data for an academic demo only.
- Each snippet should be plausible but invented.
- Return ONLY valid JSON with no markdown, code fences, or extra commentary.

The JSON object must contain exactly one key:
- news (array of exactly 3 objects)

Each news object must contain exactly these keys:
- headline (string)
- date (string, e.g. "2026-08-15")
- sentiment (string, must be exactly one of: positive, neutral, negative)
- body (string, 3-4 sentences)

The 3 news items must include exactly one positive, one neutral, and one negative sentiment.
"""
