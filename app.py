import streamlit as st
from qa_pipeline import FinancialQASystem
import time
import re
import base64

@st.cache_resource
def load_system():
    return FinancialQASystem()

@st.cache_resource
def get_base64_bg(file_path="background.jpg"):
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpg;base64,{encoded}"

st.set_page_config(
    page_title="CSV QA Agent",
    layout="wide",
)

bg_image = get_base64_bg("background.jpg")

custom_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background-image: url('{bg_image}');
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    font-family: 'Inter', sans-serif;
    color: #1e293b;
}}

[data-testid="stHeader"] {{
    background-color: rgba(255, 255, 255, 0);
}}

h1, h2, h3, h4, h5, h6 {{
    color: #0f172a;
    font-weight: 600;
    letter-spacing: 0.5px;
}}

.stTextInput > label, .stButton > button {{
    color: #334155;
}}

.stTextInput > div > input {{
    background-color: #f1f5f9;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    padding: 0.75rem;
    border-radius: 0.5rem;
}}

.stButton > button {{
    background-color: #2563eb;
    border: none;
    border-radius: 0.5rem;
    padding: 0.5rem 1.2rem;
    color: white;
    font-weight: 600;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

status_box = st.empty()

with status_box.container():
    st.markdown("🔄 **Initializing system...**")
    time.sleep(1.2)
status_box.empty()

with status_box.container():
    st.markdown("📥 **Loading QA model...**")
    time.sleep(1.2)
status_box.empty()

qa = load_system()

with status_box.container():
    st.markdown("⚙️ **Loading ML prediction models...**")
    time.sleep(1.2)
status_box.empty()

st.title("📈 Business Intelligence Q&A Console")
q = st.text_input("Ask your financial question:", placeholder="e.g., What is the revenue of Faizan Ali Khan in Q3?")

def format_answer(text):
    def indian_format(n):
        s = str(int(float(n)))
        if len(s) <= 3:
            return s
        r = s[-3:]
        s = s[:-3]
        while len(s) > 2:
            r = s[-2:] + "," + r
            s = s[:-2]
        if s:
            r = s + "," + r
        return r

    def replace_amounts(match):
        val = match.group()
        if "%" in val or len(val) < 4:
            return val
        try:
            num = float(val.replace(",", ""))
            if num >= 1000:
                return f"Rs {indian_format(num)}"
            else:
                return val
        except:
            return val

    text = re.sub(r'\b(Q[1-5])\b', r'\1_TEMP', text)
    text = re.sub(r'(?<!\d)(\d{4,})(?:\.\d+)?(?!\d)', replace_amounts, text)
    text = text.replace("_TEMP", "")
    text = text.replace("Q5", "next year Q1").replace("q5", "next year Q1")
    return text

if st.button("Submit") and q:
    with st.spinner("🔍 Analyzing your query..."):
        result = qa.answer_query(q)
        formatted_result = format_answer(result)

        lower_result = formatted_result.lower()
        color = "red" if any(word in lower_result for word in ["decrease", "decreased"]) else "green"

        st.markdown(
            f"<div class='answer' style='font-size: 1.1rem; padding: 0.75rem 1rem; margin-top: 1rem; background-color: rgba(255,255,255,0.45); color: #0f172a; border-radius: 0.25rem; border-right: 5px solid {color};'>"
            f"Answer: {formatted_result}</div>",
            unsafe_allow_html=True
        )
