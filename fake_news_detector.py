"""
Fake News Detector
-------------------
A Streamlit app that uses Gemini API to analyze news articles and detect
whether they are real or fake, with detailed reasoning and credibility score.

Run with: streamlit run app.py
"""

import streamlit as st
from google import genai
import json
import re
import os

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
)

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

default_key = os.environ.get("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=default_key,
    type="password",
)

model_name = st.sidebar.selectbox(
    "Gemini Model",
    ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**How it works:**")
st.sidebar.markdown("1. Paste any news article or headline")
st.sidebar.markdown("2. AI analyzes language, claims, sources")
st.sidebar.markdown("3. Get credibility score + detailed reasoning")
st.sidebar.markdown("4. See red flags or trust signals")

# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------

def get_client(key: str) -> genai.Client:
    return genai.Client(api_key=key)


def _safe_json_parse(raw: str, default):
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        return default


def analyze_news(client, model_name: str, news_text: str) -> dict:
    prompt = f"""You are an expert fact-checker and journalist with 20 years of experience detecting misinformation.

Analyze the following news article/headline carefully and determine if it is REAL or FAKE news.

Analyze these aspects:
1. Language patterns (sensational words, clickbait, emotional manipulation)
2. Factual claims (verifiable facts vs unverifiable claims)
3. Source credibility signals (named sources, quotes, data)
4. Logical consistency (does the story make sense?)
5. Writing quality (professional journalism vs propaganda style)
6. Red flags (all caps, excessive punctuation, conspiracy language)

Return ONLY valid JSON in exactly this format:
{{
  "verdict": "REAL" or "FAKE" or "MISLEADING" or "UNVERIFIABLE",
  "credibility_score": <number from 0 to 100>,
  "confidence": "High" or "Medium" or "Low",
  "summary": "One line verdict explanation",
  "red_flags": ["flag 1", "flag 2"],
  "trust_signals": ["signal 1", "signal 2"],
  "language_analysis": "Analysis of writing style and language used",
  "claim_analysis": "Analysis of the factual claims made",
  "advice": "What the reader should do (verify, ignore, share carefully, etc.)"
}}

NEWS TO ANALYZE:
\"\"\"
{news_text}
\"\"\"
"""
    response = client.models.generate_content(model=model_name, contents=prompt)
    return _safe_json_parse(response.text, default={
        "verdict": "UNVERIFIABLE",
        "credibility_score": 50,
        "confidence": "Low",
        "summary": "Could not analyze properly",
        "red_flags": [],
        "trust_signals": [],
        "language_analysis": "",
        "claim_analysis": "",
        "advice": "Please try again"
    })


def get_verdict_color(verdict: str) -> str:
    colors = {
        "REAL": "#00C851",
        "FAKE": "#ff4444",
        "MISLEADING": "#ffbb33",
        "UNVERIFIABLE": "#aaaaaa",
    }
    return colors.get(verdict, "#aaaaaa")


def get_score_color(score: int) -> str:
    if score >= 70:
        return "#00C851"
    elif score >= 40:
        return "#ffbb33"
    else:
        return "#ff4444"


# --------------------------------------------------------------------------
# MAIN UI
# --------------------------------------------------------------------------
st.title("🔍 Fake News Detector")
st.caption("AI-powered news credibility analyzer — paste any news and get instant fact-check analysis.")

st.markdown("---")

# Input section
st.header("📰 News Input")

input_type = st.radio(
    "What do you want to analyze?",
    ["News Headline only", "Full News Article"],
    horizontal=True,
)

if input_type == "News Headline only":
    news_input = st.text_input(
        "Paste the news headline here",
        placeholder="e.g. 'Scientists discover cure for all cancers using common household item'",
    )
else:
    news_input = st.text_area(
        "Paste the full news article here",
        placeholder="Paste the complete news article text here...",
        height=200,
    )

# Example buttons
st.caption("Or try an example:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📌 Fake Example", use_container_width=True):
        st.session_state.example = "SHOCKING!! Bill Gates ADMITS vaccines contain microchips to control population!!! Share before they DELETE this!!!"
with col2:
    if st.button("📌 Real Example", use_container_width=True):
        st.session_state.example = "India's GDP grew by 6.8% in the third quarter of 2024, according to data released by the Ministry of Statistics. The growth was primarily driven by the manufacturing and services sectors, economists said."
with col3:
    if st.button("📌 Misleading Example", use_container_width=True):
        st.session_state.example = "Coffee KILLS cancer cells according to new study. Doctors don't want you to know this secret!"

# Use example if selected
if "example" in st.session_state:
    news_input = st.session_state.example
    del st.session_state.example
    st.rerun()

# Session state
if "analysis" not in st.session_state:
    st.session_state.analysis = None

# Analyze button
analyze_btn = st.button("🔍 Analyze This News", type="primary", use_container_width=True)

if analyze_btn:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not news_input or len(news_input.strip()) < 10:
        st.error("Please enter some news text to analyze (at least 10 characters).")
    else:
        try:
            client = get_client(api_key)
            with st.spinner("Analyzing news credibility... 🔍"):
                st.session_state.analysis = analyze_news(client, model_name, news_input)
            st.success("Analysis complete!")
        except Exception as e:
            st.error(f"Error: {e}")

# --------------------------------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------------------------------
if st.session_state.analysis:
    result = st.session_state.analysis
    verdict = result.get("verdict", "UNVERIFIABLE")
    score = result.get("credibility_score", 50)
    confidence = result.get("confidence", "Low")

    st.markdown("---")
    st.header("📊 Analysis Results")

    # Top verdict + score row
    col1, col2, col3 = st.columns(3)

    with col1:
        verdict_color = get_verdict_color(verdict)
        st.markdown(
            f"""
            <div style="background-color: {verdict_color}22; border: 2px solid {verdict_color};
            border-radius: 10px; padding: 20px; text-align: center;">
                <h1 style="color: {verdict_color}; margin: 0;">{verdict}</h1>
                <p style="margin: 5px 0 0 0; color: grey;">Verdict</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        score_color = get_score_color(score)
        st.markdown(
            f"""
            <div style="background-color: {score_color}22; border: 2px solid {score_color};
            border-radius: 10px; padding: 20px; text-align: center;">
                <h1 style="color: {score_color}; margin: 0;">{score}/100</h1>
                <p style="margin: 5px 0 0 0; color: grey;">Credibility Score</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="background-color: #ffffff11; border: 2px solid #888;
            border-radius: 10px; padding: 20px; text-align: center;">
                <h1 style="margin: 0;">{"🟢" if confidence == "High" else "🟡" if confidence == "Medium" else "🔴"}</h1>
                <p style="margin: 5px 0 0 0; color: grey;">{confidence} Confidence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary
    st.info(f"**Summary:** {result.get('summary', '')}")

    # Two column analysis
    col1, col2 = st.columns(2)

    with col1:
        # Red flags
        red_flags = result.get("red_flags", [])
        if red_flags:
            st.subheader("🚩 Red Flags Detected")
            for flag in red_flags:
                st.markdown(f"- ❌ {flag}")
        else:
            st.subheader("🚩 Red Flags")
            st.markdown("✅ No major red flags detected")

    with col2:
        # Trust signals
        trust_signals = result.get("trust_signals", [])
        if trust_signals:
            st.subheader("✅ Trust Signals")
            for signal in trust_signals:
                st.markdown(f"- ✅ {signal}")
        else:
            st.subheader("✅ Trust Signals")
            st.markdown("❌ No trust signals found")

    st.markdown("---")

    # Detailed analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Language Analysis")
        st.write(result.get("language_analysis", ""))

    with col2:
        st.subheader("🔎 Claim Analysis")
        st.write(result.get("claim_analysis", ""))

    st.markdown("---")

    # Advice box
    st.subheader("💡 What Should You Do?")
    st.warning(f"**Advice:** {result.get('advice', '')}")

    # Disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This tool uses AI to analyze news patterns and language. "
        "It is not 100% accurate and should not be the sole basis for determining truth. "
        "Always verify important news from multiple trusted sources."
    )
