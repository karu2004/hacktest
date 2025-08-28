import streamlit as st
import subprocess
import time

#st.set_page_config(page_title="VNR GPT", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0rem !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .viewerBadge_link__qRIco {display: none !important;}
    .stDeployButton {display: none !important;}
    """,
    unsafe_allow_html=True
)



st.image("header.png", width=170)
st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<h1>VNR GPT</h1>", unsafe_allow_html=True)
st.markdown("""
    <span style="font-size:20px;">
        <b>PrivateGPT</b> for <b>VNRVJIET</b> – Enabling <b>students</b> and <b>faculty</b> to effortlessly search & access <b>college-related information.</b>
    </span>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Ask me anything:", unsafe_allow_html=True)
with st.form("chat_form"):
    user_query = st.text_input("", key="user_input", placeholder="Type your query...")
    submitted = st.form_submit_button("Submit")

    if submitted:
        if user_query.strip():
            with st.spinner("Analyzing... "): 
                try:
                    result = subprocess.run(["python3", "run_gpt.py", user_query], capture_output=True, text=True)
                    st.markdown(f"{result.stdout.strip()}")

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question.")


