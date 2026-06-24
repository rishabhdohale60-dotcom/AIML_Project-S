"""
AI Notes Summarizer + Quiz Generator
-------------------------------------
A Streamlit app that:
1. Accepts a PDF upload of student notes
2. Extracts text using PyPDF2
3. Uses Gemini API to generate a bullet-point summary + keywords
4. Uses Gemini API to generate 10 MCQs (with answers) from the same content
5. Lets the user download everything as a PDF report

Run with:  streamlit run app.py
"""

import streamlit as st
import PyPDF2
from google import genai
import json
import re
import os
from fpdf import FPDF
import io

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Notes Summarizer + Quiz Generator",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------------------------------
# SIDEBAR - API KEY SETUP
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

# Prefer an environment variable if set, otherwise let the user paste a key.
default_key = os.environ.get("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input(
    "Gemini API Key",
    value=default_key,
    type="password",
    help="Get a free key from https://aistudio.google.com/apikey",
)

model_name = st.sidebar.selectbox(
    "Gemini Model",
    ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
    index=0,
    help="gemini-2.5-flash is fast and free-tier friendly. Use 2.5-pro for higher quality on long notes.",
)

num_questions = st.sidebar.slider("Number of MCQs", min_value=5, max_value=20, value=10)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How to get a free Gemini API key:**\n"
    "1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)\n"
    "2. Sign in with Google\n"
    "3. Click 'Create API key'\n"
    "4. Paste it above"
)

# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file using PyPDF2."""
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def get_gemini_client(key: str) -> genai.Client:
    """Create a Gemini client. Works with both legacy (AIzaSy...) and new (AQ....) key formats."""
    return genai.Client(api_key=key)


def generate_summary_and_keywords(client: genai.Client, text: str, model_name: str) -> dict:
    """
    Ask Gemini for a structured JSON response containing:
    - summary: list of bullet point strings
    - keywords: list of important terms
    """
    prompt = f"""You are an expert academic assistant helping a student revise.

Read the following notes content and produce:
1. A concise summary as bullet points (8-12 bullets max), covering only the
   most important concepts. Keep each bullet short (1-2 sentences).
2. A list of 8-15 important keywords/terms from the content.

Return ONLY valid JSON in exactly this format, with no markdown fences,
no extra commentary, and no trailing commas:

{{
  "summary": ["point 1", "point 2", "..."],
  "keywords": ["term 1", "term 2", "..."]
}}

NOTES CONTENT:
\"\"\"
{text[:30000]}
\"\"\"
"""

    response = client.models.generate_content(model=model_name, contents=prompt)
    raw = response.text.strip()
    return _safe_json_parse(raw, default={"summary": [], "keywords": []})


def generate_quiz(client: genai.Client, text: str, model_name: str, n_questions: int) -> list:
    """
    Ask Gemini for n_questions MCQs based on the notes content.
    Returns a list of dicts: {question, options: {A,B,C,D}, answer}
    """
    prompt = f"""You are an expert exam-question setter.

Based on the notes content below, generate exactly {n_questions} multiple
choice questions (MCQs) that test understanding of the key concepts.

Rules:
- Each question must have exactly 4 options (A, B, C, D)
- Only one option should be correct
- Vary difficulty (mix of easy, medium, slightly tricky)
- Do NOT repeat the same concept in multiple questions if avoidable
- Base questions ONLY on the content given, do not invent unrelated facts

Return ONLY valid JSON, no markdown fences, no commentary, in exactly this format:

{{
  "questions": [
    {{
      "question": "What is ...?",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A"
    }}
  ]
}}

NOTES CONTENT:
\"\"\"
{text[:30000]}
\"\"\"
"""

    response = client.models.generate_content(model=model_name, contents=prompt)
    raw = response.text.strip()
    parsed = _safe_json_parse(raw, default={"questions": []})
    return parsed.get("questions", [])


def _safe_json_parse(raw: str, default):
    """
    Gemini sometimes wraps JSON in ```json ... ``` fences despite instructions.
    Strip those, then attempt to parse. Fall back to a default value on failure.
    """
    cleaned = raw.strip()
    # Remove markdown code fences if present
    cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to locate the first { and last } as a last resort
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        st.warning("⚠️ Couldn't fully parse the AI response. Showing raw output below.")
        st.code(raw)
        return default


def build_pdf_report(summary: list, keywords: list, questions: list) -> bytes:
    """Build a downloadable PDF report containing summary, keywords, and quiz+answers."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "AI Notes Summary & Quiz Report", ln=True)
    pdf.ln(2)

    # Summary section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for point in summary:
        pdf.multi_cell(0, 7, f"- {_clean_text(point)}")
    pdf.ln(3)

    # Keywords section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Keywords", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, ", ".join(_clean_text(k) for k in keywords))
    pdf.ln(3)

    # Quiz section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Quiz", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for i, q in enumerate(questions, start=1):
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 7, f"Q{i}. {_clean_text(q.get('question', ''))}")
        pdf.set_font("Helvetica", "", 11)
        opts = q.get("options", {})
        for letter in ["A", "B", "C", "D"]:
            if letter in opts:
                pdf.multi_cell(0, 6, f"   {letter}) {_clean_text(opts[letter])}")
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 7, f"   Answer: {q.get('answer', '')}", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(2)

    # fpdf2 returns bytearray; convert to bytes for Streamlit download_button
        raw_output = pdf.output()
        if isinstance(raw_output, (bytes, bytearray)):
            return bytes(raw_output)
        return raw_output.encode("latin-1")


def _clean_text(s: str) -> str:
    """FPDF's built-in Helvetica font only supports Latin-1. Replace unsupported chars."""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("latin-1", errors="replace").decode("latin-1")


# --------------------------------------------------------------------------
# MAIN APP UI
# --------------------------------------------------------------------------
st.title("📚 AI Notes Summarizer + Quiz Generator")
st.caption("Upload your notes PDF → get a quick summary, keywords, and a self-test quiz.")

uploaded_file = st.file_uploader("Upload your notes (PDF)", type=["pdf"])

# Session state to persist results across reruns (e.g. when quiz radio buttons change)
if "summary_data" not in st.session_state:
    st.session_state.summary_data = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

col1, col2 = st.columns(2)
with col1:
    generate_clicked = st.button("🚀 Generate Summary + Quiz", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("🔄 Clear Results", use_container_width=True)

if clear_clicked:
    st.session_state.summary_data = None
    st.session_state.quiz_data = None
    st.session_state.extracted_text = ""
    st.rerun()

if generate_clicked:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar first.")
    elif not uploaded_file:
        st.error("Please upload a PDF file first.")
    else:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.extracted_text = text

        if not text or len(text) < 30:
            st.error("Couldn't extract readable text from this PDF. It may be a scanned/image-based PDF.")
        else:
            try:
                client = get_gemini_client(api_key)

                with st.spinner("Generating summary and keywords with Gemini..."):
                    st.session_state.summary_data = generate_summary_and_keywords(client, text, model_name)

                with st.spinner(f"Generating {num_questions} quiz questions with Gemini..."):
                    st.session_state.quiz_data = generate_quiz(client, text, model_name, num_questions)

                st.success("Done! Scroll down to see your summary and quiz.")
            except Exception as e:
                st.error(f"Something went wrong calling the Gemini API: {e}")

# --------------------------------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------------------------------
if st.session_state.summary_data:
    st.markdown("---")
    st.header("📝 Summary")
    summary_points = st.session_state.summary_data.get("summary", [])
    if summary_points:
        for point in summary_points:
            st.markdown(f"- {point}")
    else:
        st.info("No summary points were generated.")

    st.header("🔑 Keywords")
    keywords = st.session_state.summary_data.get("keywords", [])
    if keywords:
        st.write(" | ".join(f"`{k}`" for k in keywords))
    else:
        st.info("No keywords were generated.")

if st.session_state.quiz_data:
    st.markdown("---")
    st.header("🧠 Quiz")
    st.caption("Select your answers, then click 'Check Answers' at the bottom.")

    user_answers = {}
    for i, q in enumerate(st.session_state.quiz_data, start=1):
        st.subheader(f"Q{i}. {q.get('question', '')}")
        opts = q.get("options", {})
        labels = [f"{letter}) {opts[letter]}" for letter in ["A", "B", "C", "D"] if letter in opts]
        choice = st.radio(
            f"Select answer for Q{i}",
            labels,
            index=None,
            key=f"q_{i}",
            label_visibility="collapsed",
        )
        user_answers[i] = choice

    if st.button("✅ Check Answers"):
        score = 0
        total = len(st.session_state.quiz_data)
        for i, q in enumerate(st.session_state.quiz_data, start=1):
            correct_letter = q.get("answer", "")
            correct_text = q.get("options", {}).get(correct_letter, "")
            chosen = user_answers.get(i)
            is_correct = chosen is not None and chosen.startswith(f"{correct_letter})")
            if is_correct:
                score += 1
                st.success(f"Q{i}: Correct ✅ ({correct_letter}) {correct_text}")
            else:
                st.error(f"Q{i}: Incorrect ❌ — Correct answer: {correct_letter}) {correct_text}")
        st.info(f"**Your Score: {score} / {total}**")

    # PDF download
    st.markdown("---")
    pdf_bytes = build_pdf_report(
        st.session_state.summary_data.get("summary", []),
        st.session_state.summary_data.get("keywords", []),
        st.session_state.quiz_data,
    )
    st.download_button(
        "⬇️ Download Summary + Quiz as PDF",
        data=pdf_bytes,
        file_name="notes_summary_quiz.pdf",
        mime="application/pdf",
    )
