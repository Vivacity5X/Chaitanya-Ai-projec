import streamlit as st
import os
from transcribe import transcribe_audio
from text_processing import summarize_text, extract_action_items
from transformers import pipeline

# --- 1. Page Configuration (MUST be the first st command) ---
st.set_page_config(
    page_title="AI Meeting Summarizer",
    page_icon="🚀",  # <--- Set a fun emoji icon
    layout="wide"  # <--- Use the full page width
)

# --- 2. Cache the Models ---
@st.cache_resource
def load_models():
    """Loads the AI models once and caches them."""
    print("Loading AI models... (This should only run once!)")
    # Use the SMALLER model for better performance
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Models loaded.")
    return summarizer, classifier

# Load the models
summarizer_model, classifier_model = load_models()


# --- 3. Sidebar for Controls ---
with st.sidebar:
    st.header("🚀 AI Meeting Summarizer")
    st.write("Upload a meeting audio file and get a summary, action items, and the full transcript.")
    
    uploaded_file = st.file_uploader("Upload your audio file (.mp3, .wav)", type=["mp3", "wav"])

    if uploaded_file is not None:
        # Save the file temporarily
        temp_file_path = os.path.join("temp_audio_file." + uploaded_file.name.split('.')[-1])
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(temp_file_path, format=uploaded_file.type)

        # --- The 'Generate' Button ---
        if st.button("Generate Summary & Action Items", type="primary"):
            try:
                # Step A: Transcribe
                with st.spinner("Step 1/3: Transcribing audio..."):
                    transcript = transcribe_audio(temp_file_path)
                
                if "Error:" in transcript:
                    st.error(transcript)
                else:
                    # Step B: Summarize
                    with st.spinner("Step 2/3: Generating summary..."):
                        summary = summarize_text(summarizer_model, transcript)
                    
                    # Step C: Extract Action Items
                    with st.spinner("Step 3/3: Extracting action items..."):
                        action_items = extract_action_items(classifier_model, transcript)
                    
                    # --- Save results to Session State ---
                    st.session_state.summary = summary
                    st.session_state.action_items = action_items
                    st.session_state.transcript = transcript
                    
                    st.success("Processing Complete!")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            finally:
                # Clean up the temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    
    st.divider()
    st.caption("A project by Aryan Singh. Powered by Streamlit, Whisper, and Hugging Face.")


# --- 4. Main Page for Results ---
st.title("Your Meeting Report")

# Check if results exist in session state
if "summary" in st.session_state:
    
    # --- Two-Column Layout ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Summary")
        with st.container(border=True, height=250):
            st.write(st.session_state.summary)

    with col2:
        st.subheader("📌 Action Items")
        with st.container(border=True, height=250):
            if st.session_state.action_items:
                for item in st.session_state.action_items:
                    st.write(f"- {item}.")
            else:
                st.info("No specific action items were detected.")

    st.divider()

    # --- Expander for Full Transcript ---
    with st.expander("📜 View Full Transcript"):
        st.text_area("Transcript", st.session_state.transcript, height=300)

else:
    # --- Initial State (before button is pressed) ---
    st.info("Please upload an audio file and click 'Generate' in the sidebar to see your results.")
