import streamlit as st
import os
import subprocess

st.set_page_config(page_title="Knowledge Base Admin", layout="wide")

st.markdown("""
    <style>
        .block-container {
        padding-top: 0rem !important;
        }
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .viewerBadge_link__qRIco {display: none !important;}
        .stDeployButton {display: none !important;}
        .file-list {
            max-width: 600px;
            margin: auto;
        }
        .delete-button {
            color: red;
            font-weight: bold;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

st.image("header.png", width=170) 

KB_DIR = os.path.join(os.getcwd(), "knowledge_base")

if not os.path.exists(KB_DIR):
    os.makedirs(KB_DIR)

st.title("Knowledge Base Admin")
st.subheader("Manage your documents")

st.markdown('<div class="upload-container">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    file_path = os.path.join(KB_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ {uploaded_file.name} uploaded successfully!")
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
files = os.listdir(KB_DIR)

if files:
    for file in files:
        col1, col2 = st.columns([4, 1])  # Adjust column width
        with col1:
            st.write(f"📜 {file}")
        with col2:
            if st.button("❌", key=file):  # Delete button
                os.remove(os.path.join(KB_DIR, file))
                st.warning(f"🗑️ Deleted: {file}")
                st.rerun() # Refresh after deletion
else:
    st.info("No PDF files found.")

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    if st.button("📥 Finalize & Ingest Documents", use_container_width=True):
        st.info("🚀 Ingesting documents, please wait...")
        try:
            result = subprocess.run(["python3", "ingest_chroma.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("✅ Documents ingested successfully!")
            else:
                st.error(f"⚠️ Ingestion failed:\n{result.stderr}")
        except Exception as e:
            st.error(f"❌ Error running ingestion script: {e}")

