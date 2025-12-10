import streamlit as st

from orchestrator import multimodal_turn, analyze_environment
from memory import get_global_memory
from logging_utils import get_logger

logger = get_logger()

st.set_page_config(page_title="Multimodal AI Assistant", layout="wide")

st.title("🧠 Multimodal AI Assistant")

st.markdown(
    """
This assistant can:
- Capture a webcam frame
- Detect objects in your environment
- Use that as context for an LLM
- Maintain short-term conversation memory
- Optionally speak out responses
"""
)

memory = get_global_memory()

# Sidebar controls
st.sidebar.header("Settings")
use_audio = st.sidebar.checkbox("Use microphone input (STT)", value=False)
if st.sidebar.button("Clear conversation memory"):
    memory.clear()
    st.sidebar.success("Memory cleared.")

# Layout
col_left, col_right = st.columns([1.2, 1.8])

with col_left:
    st.subheader("Environment")
    if st.button("Capture frame & detect objects"):
        frame, objects = analyze_environment()
        if frame is None:
            st.error("Failed to capture frame from webcam.")
        else:
            st.image(frame, channels="BGR", caption="Current view")
            if objects:
                st.success(f"Detected objects: {', '.join(objects)}")
            else:
                st.info("No notable objects detected.")

with col_right:
    st.subheader("Chat")

    # Display conversation history
    history = memory.get_history()
    for turn in history:
        role = "🧑 You" if turn["role"] == "user" else "🤖 Assistant"
        st.markdown(f"**{role}:** {turn['content']}")

    st.markdown("---")

    user_input = st.text_area("Your message:", height=80, key="user_input")

    if st.button("Send"):
        if not user_input and not use_audio:
            st.warning("Type a message or enable audio input.")
        else:
            with st.spinner("Thinking..."):
                reply, objects_used, frame = multimodal_turn(
                    user_text=user_input,
                    use_audio=use_audio
                )

            if frame is not None:
                st.image(frame, channels="BGR", caption="View used for this reply")

            st.success("Assistant replied.")
            st.rerun()
