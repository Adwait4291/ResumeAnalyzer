import streamlit as st
import os
from pathlib import Path
from typing import Optional, IO

# Import functions from your source modules
from src.nlp_processor.models import load_spacy_model, get_loaded_model_name
from src.nlp_processor.similarity import calculate_similarity, extract_and_compare_keywords
from src.utils.helpers import setup_app_logger, clean_text, read_text_file # Optional: , read_pdf_file

# --- Setup Logger ---
logger = setup_app_logger(__name__)

# --- App Configuration ---
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Load NLP Model ---
# Model name is determined in models.py based on .env or default
MODEL_NAME = get_loaded_model_name()
nlp = load_spacy_model() # Uses the model name determined in models.py

# --- Helper Function for Reading Uploaded File ---
def get_text_from_upload(uploaded_file: Optional[IO[bytes]]) -> Optional[str]:
    """Reads text from Streamlit uploaded file object."""
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name
    file_type = uploaded_file.type
    logger.info(f"Processing uploaded file: {file_name} (Type: {file_type})")

    try:
        # Simple text reading for common types
        if file_type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        # Optional: Add PDF reading (requires PyPDF2)
        # elif file_type == "application/pdf" and read_pdf_file is not None:
        #     # Need to save temp file to use path-based reader, or use BytesIO
        #     # For simplicity, let's assume direct reading if library supports bytes IO
        #     # This part needs careful implementation based on library used
        #     logger.warning("Direct PDF reading from upload stream not implemented in this example.")
        #     # return read_pdf_bytesio(uploaded_file) # Hypothetical function
        #     return None # Placeholder
        elif file_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
             st.warning(f"Direct reading of {file_type} not fully implemented in this basic version. Please paste text.")
             return None
        else:
            logger.warning(f"Unsupported file type uploaded: {file_type}")
            st.warning(f"Unsupported file type '{file_type}'. Please upload .txt or paste text.")
            return None
    except UnicodeDecodeError:
        logger.error(f"UTF-8 decoding failed for {file_name}. Trying latin-1.")
        try:
            # Reset buffer position and try again with latin-1
            uploaded_file.seek(0)
            return uploaded_file.read().decode("latin-1")
        except Exception as e:
             logger.error(f"Failed to read {file_name} with any encoding: {e}")
             st.error(f"Could not read the uploaded file '{file_name}'. Check encoding.")
             return None
    except Exception as e:
        logger.error(f"Error processing uploaded file {file_name}: {e}", exc_info=True)
        st.error(f"An error occurred while processing the uploaded file: {e}")
        return None

# --- Initialize Session State ---
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'similarity_score' not in st.session_state:
    st.session_state.similarity_score = 0.0
if 'keyword_results' not in st.session_state:
    st.session_state.keyword_results = {"common": [], "unique_text1": [], "unique_text2": []}
if 'error_message' not in st.session_state:
    st.session_state.error_message = None


# --- App UI ---
st.title("📄 Resume Analysis App")
st.markdown(f"""
Analyze how well your resume aligns with a job description.
Powered by `spaCy` (Model: `{MODEL_NAME}`).
""")

st.sidebar.header("Options")
# Add any options here later, e.g., changing similarity threshold

st.markdown("---")

# Input Areas
col1, col2 = st.columns(2)

# Store inputs outside the columns to access them later easily
resume_input_method = col1.radio("Resume Input Method:", ("Paste Text", "Upload File"), key="resume_method", horizontal=True)
jd_input_method = col2.radio("Job Desc. Input Method:", ("Paste Text",), key="jd_method", horizontal=True) # Only paste for JD for now

resume_text_area = None
uploaded_resume_file = None
jd_text_area = None

with col1:
    st.subheader("Your Resume")
    if resume_input_method == "Paste Text":
        resume_text_area = st.text_area("Paste resume text:", height=300, key="resume_paste")
    else:
        # Allow txt for now, add others if reading logic is implemented
        uploaded_resume_file = st.file_uploader(
            "Upload Resume",
            type=['txt'], # Add 'pdf', 'docx' if read_pdf/read_docx implemented
            key="resume_upload"
        )

with col2:
    st.subheader("Job Description")
    if jd_input_method == "Paste Text":
         jd_text_area = st.text_area("Paste job description:", height=300, key="jd_paste")
    # Add URL input later if needed

st.markdown("---")

# Analysis Trigger
analyze_button = st.button("Analyze Alignment", type="primary", disabled=(nlp is None))

if nlp is None:
     st.error("NLP Model not loaded. Analysis is disabled. Please check setup instructions and logs.")

# --- Analysis Logic ---
if analyze_button and nlp:
    st.session_state.analysis_complete = False # Reset state on new analysis
    st.session_state.error_message = None
    resume_content = None
    jd_content = None

    # Get Resume Content
    if resume_input_method == "Paste Text":
        resume_content = resume_text_area
    else:
        resume_content = get_text_from_upload(uploaded_resume_file)

    # Get Job Description Content
    if jd_input_method == "Paste Text":
        jd_content = jd_text_area

    # Validate inputs
    if not resume_content or not jd_content:
        st.warning("Please provide content for both the resume and the job description.")
        st.session_state.error_message = "Missing input."
    else:
        # Clean the inputs
        cleaned_resume = clean_text(resume_content)
        cleaned_jd = clean_text(jd_content)

        if not cleaned_resume or not cleaned_jd:
             st.warning("After cleaning, one or both inputs are empty.")
             st.session_state.error_message = "Empty input after cleaning."
        else:
            logger.info("Starting analysis...")
            with st.spinner("Analyzing alignment..."):
                try:
                    # Calculate similarity
                    st.session_state.similarity_score = calculate_similarity(nlp, cleaned_resume, cleaned_jd)

                    # Extract keywords
                    st.session_state.keyword_results = extract_and_compare_keywords(nlp, cleaned_resume, cleaned_jd)

                    st.session_state.analysis_complete = True
                    logger.info("Analysis complete.")

                except Exception as e:
                    logger.error(f"An error occurred during the analysis process: {e}", exc_info=True)
                    st.error(f"An unexpected error occurred during analysis: {e}")
                    st.session_state.error_message = str(e)


# --- Display Results ---
if st.session_state.analysis_complete:
    st.subheader("📊 Analysis Results")

    score = st.session_state.similarity_score
    st.metric(label="Alignment Score", value=f"{score:.1%}") # Display as percentage

    # Simple Interpretation based on score
    interpretation = "Needs Improvement"
    color = "red"
    if score > 0.90:
        interpretation = "Excellent!"
        color = "blue"
    elif score > 0.80:
        interpretation = "Good"
        color = "green"
    elif score > 0.70:
        interpretation = "Moderate"
        color = "orange"

    st.markdown(f"**Interpretation:** <span style='color:{color};'>{interpretation}</span>", unsafe_allow_html=True)
    st.progress(score)

    # Display Keywords
    st.markdown("---")
    st.subheader("🔑 Keyword Analysis")
    keywords = st.session_state.keyword_results
    if not keywords['common'] and not keywords['unique_text1'] and not keywords['unique_text2']:
         st.info("Could not extract keywords (check input text length/content or logs).")
    else:
        kw_col1, kw_col2 = st.columns(2)
        with kw_col1:
            st.info("**Common Keywords**")
            if keywords['common']:
                 st.write(", ".join(f"`{kw}`" for kw in keywords['common']))
            else:
                 st.write("_None found_")

            st.warning("**Keywords in JD (Not in Resume)**")
            if keywords['unique_text2']:
                 st.write(", ".join(f"`{kw}`" for kw in keywords['unique_text2']))
            else:
                 st.write("_None found_")

        with kw_col2:
            st.success("**Keywords in Resume (Maybe not in JD)**") # Wording might need adjustment
            if keywords['unique_text1']:
                 st.write(", ".join(f"`{kw}`" for kw in keywords['unique_text1']))
            else:
                 st.write("_None found_")

        st.caption("Keywords based on common Nouns & Proper Nouns (lemmatized, lowercased).")


elif st.session_state.error_message:
     st.error(f"Analysis could not be completed: {st.session_state.error_message}")


# --- Footer ---
st.markdown("---")
st.caption("Built by [Your Name/Team] | Using Streamlit, spaCy")