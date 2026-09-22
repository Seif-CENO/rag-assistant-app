import streamlit as st
import requests
import os
from dotenv import load_dotenv


load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL")

st.set_page_config(page_title="OSHA Safety Assistant", page_icon="👷", layout="centered")
st.title("👷 OSHA Safety & PPE Auditor")
st.write("Upload an image of a worksite and ask a question to verify safety compliance.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "detections" in message and message["detections"]:
            st.caption(f"**YOLO Detections:** {', '.join(message['detections'])}")
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")

uploaded_file = st.file_uploader("Upload a worksite image (Optional)", type=["jpg", "jpeg", "png"])
prompt = st.chat_input("Ask a safety question...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing rules and imagery..."):
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
                    st.caption(f"**YOLO Detections:** {', '.join(detections)}")
                
                if sources:
                    with st.expander("View Sources"):
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