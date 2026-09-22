import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="OSHA Auditor", page_icon="🛡️", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .css-1d391kg { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)
st.title("OSHA Workplace Safety Auditor")
st.markdown("Automated compliance verification using YOLOv8 PPE detection and RAG-powered OSHA guidelines.")
st.markdown("---")

col1, col2 = st.columns([1, 2], gap="large")
with col1:
    st.subheader("Site Inspection")
    uploaded_file = st.file_uploader("Upload Worksite Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        st.image(uploaded_file, caption="Live Telemetry Feed", use_column_width=True)
    else:
        st.info("Awaiting visual input. You can still ask text-only safety questions in the chat.")

with col2:
    st.subheader("Compliance Audit Log")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("detections"):
                st.caption(f"**Vision System:** {', '.join(message['detections'])}")
            if message.get("sources"):
                with st.expander("View OSHA Citations"):
                    for source in message["sources"]:
                        st.write(f"- {source}")

    prompt = st.chat_input("Enter safety query (e.g., 'Are there any PPE violations here?')")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing visual telemetry and querying regulatory database..."):
                try:
                    if uploaded_file is not None:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {"question": prompt}
                        response = requests.post(f"{API_BASE_URL}/query-vision", data=data, files=files)
                    else:
                        json_data = {"question": prompt}
                        response = requests.post(f"{API_BASE_URL}/query", json=json_data)

                    response.raise_for_status()
                    result = response.json()
                    answer = result.get("answer", "Error generating answer.")
                    sources = result.get("sources", [])
                    detections = result.get("detections", [])

                    st.markdown(answer)
                    if detections:
                        st.caption(f"**Vision System:** {', '.join(detections)}")
                    
                    if sources:
                        with st.expander("View OSHA Citations"):
                            for source in sources:
                                st.write(f"- {source}")

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources,
                        "detections": detections
                    })
                except requests.exceptions.RequestException as e:
                    st.error(f"Backend API Error: {e}")