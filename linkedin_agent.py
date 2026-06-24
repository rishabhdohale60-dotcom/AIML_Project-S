"""
LinkedIn AI Post Agent
-----------------------
A Streamlit app that acts as your personal LinkedIn content agent.
It helps AIML students:
1. Generate post ideas based on their interests/projects
2. Create a weekly posting schedule
3. Write ready-to-copy LinkedIn posts

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
    page_title="LinkedIn AI Post Agent",
    page_icon="💼",
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
    help="Same key as your Notes Summarizer project",
)

model_name = st.sidebar.selectbox(
    "Gemini Model",
    ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips for best results:**")
st.sidebar.markdown("- Be specific about your projects")
st.sidebar.markdown("- Mention your tech stack")
st.sidebar.markdown("- Include your goals (internship, job, etc.)")

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


def generate_post_ideas(client, model_name: str, profile: dict) -> list:
    prompt = f"""You are a LinkedIn content strategist helping a 2nd year AIML (Artificial Intelligence & Machine Learning) student build their personal brand on LinkedIn.

Student Profile:
- Name: {profile['name']}
- College Year: 2nd Year AIML Student
- Recent Projects: {profile['projects']}
- Skills: {profile['skills']}
- Goals: {profile['goals']}
- Interests: {profile['interests']}

Generate exactly 7 creative and engaging LinkedIn post IDEAS for this student — one for each day of the week.
Each idea should be specific, actionable, and relevant to AIML students.

Mix these post types:
- Project showcase (what you built + learnings)
- Concept explained simply (teach something AI/ML related)
- Personal journey / struggle + lesson
- Tool or resource recommendation
- Behind the scenes (your coding process)
- Achievement or milestone
- Question / poll to engage audience

Return ONLY valid JSON in exactly this format:
{{
  "ideas": [
    {{
      "day": "Monday",
      "type": "Project Showcase",
      "topic": "Short specific topic here",
      "hook": "First attention-grabbing line of the post",
      "why": "Why this post will get engagement"
    }}
  ]
}}"""

    response = client.models.generate_content(model=model_name, contents=prompt)
    parsed = _safe_json_parse(response.text, default={"ideas": []})
    return parsed.get("ideas", [])


def generate_full_post(client, model_name: str, idea: dict, profile: dict) -> str:
    prompt = f"""You are a LinkedIn content writer helping a 2nd year AIML student write an engaging post.

Student Profile:
- Name: {profile['name']}
- Projects: {profile['projects']}
- Skills: {profile['skills']}
- Goals: {profile['goals']}

Write a complete, ready-to-post LinkedIn post based on this idea:
- Day: {idea['day']}
- Type: {idea['type']}
- Topic: {idea['topic']}
- Hook (first line): {idea['hook']}

LinkedIn Post Writing Rules:
1. Start with the hook line (attention grabbing, no "I" as first word)
2. Keep paragraphs SHORT (1-2 lines max) with line breaks between them
3. Use emojis naturally (not too many, not too few — 3-5 max)
4. Include a personal story or specific detail to make it authentic
5. End with a clear call-to-action (question to readers OR "Follow for more")
6. Add 3-5 relevant hashtags at the end
7. Total length: 150-250 words (ideal LinkedIn length)
8. Tone: Friendly, genuine, student perspective — NOT corporate or fake

Write ONLY the post content, nothing else. No titles, no explanations."""

    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text.strip()


def generate_monthly_schedule(client, model_name: str, profile: dict) -> dict:
    prompt = f"""You are a LinkedIn content strategist for a 2nd year AIML student.

Student Profile:
- Projects: {profile['projects']}
- Skills: {profile['skills']}
- Goals: {profile['goals']}

Create a 4-week LinkedIn posting schedule (3 posts per week — Monday, Wednesday, Friday).
Make it strategic — build from basics to advanced, mix personal and technical content.

Return ONLY valid JSON:
{{
  "schedule": [
    {{
      "week": 1,
      "theme": "Week theme here",
      "posts": [
        {{"day": "Monday", "type": "Post type", "topic": "Specific topic"}},
        {{"day": "Wednesday", "type": "Post type", "topic": "Specific topic"}},
        {{"day": "Friday", "type": "Post type", "topic": "Specific topic"}}
      ]
    }}
  ]
}}"""

    response = client.models.generate_content(model=model_name, contents=prompt)
    parsed = _safe_json_parse(response.text, default={"schedule": []})
    return parsed.get("schedule", [])


# --------------------------------------------------------------------------
# MAIN UI
# --------------------------------------------------------------------------
st.title("💼 LinkedIn AI Post Agent")
st.caption("Aapka personal LinkedIn content manager — ideas, schedule, aur ready-to-post content.")

st.markdown("---")

# Profile Input Section
st.header("📋 Apna Profile Bataiye")
st.caption("Jitna detail doge, utna better post banega.")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Aapka naam", placeholder="Rishabh Dohale")
    projects = st.text_area(
        "Recent Projects (comma separated)",
        placeholder="AI Notes Summarizer, LinkedIn Agent, Rock Paper Scissors game",
        height=100,
    )
    skills = st.text_area(
        "Your Skills",
        placeholder="Python, Streamlit, Gemini API, Machine Learning, Git, GitHub",
        height=100,
    )

with col2:
    goals = st.text_input(
        "Aapka LinkedIn Goal",
        placeholder="Internship dhundna, network banana, projects showcase karna",
    )
    interests = st.text_area(
        "AIML Topics jo aapko pasand hain",
        placeholder="Generative AI, NLP, Computer Vision, LLMs, Prompt Engineering",
        height=100,
    )

profile = {
    "name": name or "Student",
    "projects": projects or "AI/ML Projects",
    "skills": skills or "Python, Machine Learning",
    "goals": goals or "Build network and get internship",
    "interests": interests or "AI, ML, Deep Learning",
}

st.markdown("---")

# Action Buttons
st.header("🚀 Kya Karna Hai?")

col1, col2, col3 = st.columns(3)

with col1:
    ideas_btn = st.button("💡 7 Post Ideas Generate Karo\n(1 Week)", use_container_width=True, type="primary")

with col2:
    schedule_btn = st.button("📅 1 Month Schedule\nBanao", use_container_width=True)

with col3:
    st.caption("Post ideas generate karne ke baad neeche se koi bhi idea select karo aur full post likhwao 👇")

# Session state
if "ideas" not in st.session_state:
    st.session_state.ideas = []
if "schedule" not in st.session_state:
    st.session_state.schedule = []
if "generated_post" not in st.session_state:
    st.session_state.generated_post = ""
if "selected_idea" not in st.session_state:
    st.session_state.selected_idea = None

# --------------------------------------------------------------------------
# GENERATE IDEAS
# --------------------------------------------------------------------------
if ideas_btn:
    if not api_key:
        st.error("Pehle sidebar mein Gemini API key daalo.")
    else:
        try:
            client = get_client(api_key)
            with st.spinner("7 post ideas soch raha hoon aapke liye... 🤔"):
                st.session_state.ideas = generate_post_ideas(client, model_name, profile)
                st.session_state.generated_post = ""
            st.success("Ideas ready hain! Neeche dekho 👇")
        except Exception as e:
            st.error(f"Error: {e}")

# --------------------------------------------------------------------------
# GENERATE SCHEDULE
# --------------------------------------------------------------------------
if schedule_btn:
    if not api_key:
        st.error("Pehle sidebar mein Gemini API key daalo.")
    else:
        try:
            client = get_client(api_key)
            with st.spinner("1 mahine ka schedule plan kar raha hoon... 📅"):
                st.session_state.schedule = generate_monthly_schedule(client, model_name, profile)
            st.success("Schedule ready hai! Neeche dekho 👇")
        except Exception as e:
            st.error(f"Error: {e}")

# --------------------------------------------------------------------------
# DISPLAY IDEAS + FULL POST GENERATOR
# --------------------------------------------------------------------------
if st.session_state.ideas:
    st.markdown("---")
    st.header("💡 Aapke 7 Post Ideas (1 Week)")

    for i, idea in enumerate(st.session_state.ideas):
        with st.expander(f"**{idea.get('day', f'Day {i+1}')}** — {idea.get('type', '')} | {idea.get('topic', '')}"):
            st.markdown(f"**🎯 Hook (First line):** {idea.get('hook', '')}")
            st.markdown(f"**📈 Kyun kaam karega:** {idea.get('why', '')}")

            if st.button(f"✍️ Is idea ka pura post likho", key=f"write_{i}"):
                if not api_key:
                    st.error("API key daalo pehle.")
                else:
                    try:
                        client = get_client(api_key)
                        with st.spinner("Post likh raha hoon... ✍️"):
                            st.session_state.generated_post = generate_full_post(
                                client, model_name, idea, profile
                            )
                            st.session_state.selected_idea = idea
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Display generated post
    if st.session_state.generated_post:
        st.markdown("---")
        st.header("📝 Aapki Ready-to-Post LinkedIn Post")
        st.caption(f"Topic: {st.session_state.selected_idea.get('topic', '')}")

        # Show post in a nice text area for easy copying
        st.text_area(
            "Ye copy karo aur LinkedIn pe paste karo 👇",
            value=st.session_state.generated_post,
            height=350,
            key="post_display",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **Tip:** Text area pe click karo, Ctrl+A se sab select karo, phir Ctrl+C se copy karo.")
        with col2:
            # Regenerate button
            if st.button("🔄 Dobara Generate Karo (alag version)", use_container_width=True):
                if api_key and st.session_state.selected_idea:
                    try:
                        client = get_client(api_key)
                        with st.spinner("Naya version likh raha hoon..."):
                            st.session_state.generated_post = generate_full_post(
                                client, model_name, st.session_state.selected_idea, profile
                            )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

# --------------------------------------------------------------------------
# DISPLAY MONTHLY SCHEDULE
# --------------------------------------------------------------------------
if st.session_state.schedule:
    st.markdown("---")
    st.header("📅 Aapka 1 Month LinkedIn Schedule")
    st.caption("Har hafte 3 posts — Monday, Wednesday, Friday")

    for week in st.session_state.schedule:
        week_num = week.get("week", "")
        theme = week.get("theme", "")
        posts = week.get("posts", [])

        st.subheader(f"Week {week_num}: {theme}")

        cols = st.columns(3)
        for j, post in enumerate(posts):
            with cols[j]:
                st.markdown(f"**{post.get('day', '')}**")
                st.markdown(f"*{post.get('type', '')}*")
                st.markdown(post.get('topic', ''))
                st.markdown("---")
