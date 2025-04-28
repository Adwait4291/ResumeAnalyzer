import streamlit as st
import os
from pathlib import Path
from typing import Optional, IO

# Import functions from your source modules
from src.nlp_processor.models import load_spacy_model, get_loaded_model_name
from src.nlp_processor.similarity import calculate_similarity, extract_and_compare_keywords
# Import stream readers from helpers
from src.utils.helpers import (
    setup_app_logger,
    clean_text,
    read_text_from_stream,
    read_pdf_from_stream,
    read_docx_from_stream
)

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
MODEL_NAME = get_loaded_model_name()
nlp = load_spacy_model()

# --- Helper Function for Reading Uploaded File ---
def get_text_from_upload(uploaded_file: Optional[IO[bytes]]) -> Optional[str]:
    """Reads text from Streamlit uploaded file object using appropriate helper."""
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name
    file_type = uploaded_file.type # Get MIME type
    logger.info(f"Processing uploaded file: {file_name} (Type: {file_type})")

    try:
        if file_type == "text/plain":
            return read_text_from_stream(uploaded_file, file_name)
        elif file_type == "application/pdf":
            text = read_pdf_from_stream(uploaded_file, file_name)
            if text is None:
                st.error(f"Could not read PDF: '{file_name}'. Ensure PyPDF2 is installed and the file is not corrupted/password-protected.")
            return text
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
             text = read_docx_from_stream(uploaded_file, file_name)
             if text is None:
                  st.error(f"Could not read DOCX: '{file_name}'. Ensure python-docx is installed and the file is not corrupted.")
             return text
        # Add other MIME types if needed (e.g., older .doc)
        # elif file_type == "application/msword":
        #    st.warning("Reading older '.doc' files is not supported. Please convert to DOCX or TXT.")
        #    return None
        else:
            logger.warning(f"Unsupported file type uploaded: {file_type} for file {file_name}")
            st.warning(f"Unsupported file type '{file_type}'. Please upload .txt, .pdf, or .docx.")
            return None
    except Exception as e:
        logger.error(f"An unexpected error occurred processing uploaded file {file_name}: {e}", exc_info=True)
        st.error(f"An error occurred while reading the file: {e}")
        return None

# --- Initialize Session State (keep as before) ---
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

st.markdown("---")

# Input Areas
col1, col2 = st.columns(2)

resume_input_method = col1.radio("Resume Input Method:", ("Paste Text", "Upload File"), key="resume_method", horizontal=True)
jd_input_method = col2.radio("Job Desc. Input Method:", ("Paste Text",), key="jd_method", horizontal=True) # Keep JD as paste for now

resume_text_area = None
uploaded_resume_file = None
jd_text_area = None

with col1:
    st.subheader("Your Resume")
    if resume_input_method == "Paste Text":
        resume_text_area = st.text_area("Paste resume text:", height=300, key="resume_paste", placeholder="Paste your full resume text here...")
    else:
        # *** UPDATE FILE TYPES HERE ***
        uploaded_resume_file = st.file_uploader(
            "Upload Resume File",
            type=['txt', 'pdf', 'docx'], # Accept pdf and docx
            key="resume_upload",
            help="Upload your resume in .txt, .pdf, or .docx format."
        )

with col2:
    st.subheader("Job Description")
    if jd_input_method == "Paste Text":
         jd_text_area = st.text_area("Paste job description:", height=300, key="jd_paste", placeholder="Paste the full job description text here...")

st.markdown("---")

# Analysis Trigger
analyze_button = st.button("Analyze Alignment", type="primary", disabled=(nlp is None))

if nlp is None:
     st.error("NLP Model not loaded. Analysis is disabled. Please check setup instructions and logs.")

# --- Analysis Logic (Updated Input Gathering) ---
if analyze_button and nlp:
    st.session_state.analysis_complete = False
    st.session_state.error_message = None
    resume_content = None
    jd_content = None

    # Get Resume Content (Updated)
    if resume_input_method == "Paste Text":
        resume_content = resume_text_area
        if not resume_content: logger.warning("Resume paste area is empty.")
    else:
        # Process uploaded file
        if uploaded_resume_file is not None:
            resume_content = get_text_from_upload(uploaded_resume_file)
            # Error messages are now handled within get_text_from_upload
        else:
             logger.warning("File upload method selected, but no file uploaded.")

    # Get Job Description Content
    if jd_input_method == "Paste Text":
        jd_content = jd_text_area
        if not jd_content: logger.warning("Job Description paste area is empty.")

    # Validate inputs
    if not resume_content or not jd_content:
        st.warning("Please provide content for both the resume and the job description.")
        st.session_state.error_message = "Missing input."
    else:
        # Clean the inputs
        cleaned_resume = clean_text(resume_content)
        cleaned_jd = clean_text(jd_content)

        if not cleaned_resume or not cleaned_jd:
             st.warning("After cleaning, one or both inputs are empty. Analysis cannot proceed.")
             st.session_state.error_message = "Input empty after cleaning."
        else:
            logger.info("Starting analysis...")
            with st.spinner("Analyzing alignment... This may take a moment."):
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


# --- Display Results (Keep as before) ---
if st.session_state.analysis_complete:
    st.subheader("📊 Analysis Results")
    # ... (rest of the results display code remains the same) ...
    score = st.session_state.similarity_score
    st.metric(label="Alignment Score", value=f"{score:.1%}")

    interpretation = "Needs Improvement"
    color = "red"
    # ... (interpretation logic) ...
    if score > 0.90: interpretation, color = "Excellent!", "blue"
    elif score > 0.80: interpretation, color = "Good", "green"
    elif score > 0.70: interpretation, color = "Moderate", "orange"

    st.markdown(f"**Interpretation:** <span style='color:{color};'>{interpretation}</span>", unsafe_allow_html=True)
    st.progress(score)

    st.markdown("---")
    st.subheader("🔑 Keyword Analysis")
    keywords = st.session_state.keyword_results
    # ... (keyword display logic) ...
    if not keywords['common'] and not keywords['unique_text1'] and not keywords['unique_text2']:
         st.info("Could not extract keywords (check input text length/content or logs).")
    else:
        kw_col1, kw_col2 = st.columns(2)
        with kw_col1:
            st.info("**Common Keywords**")
            if keywords['common']: st.write(", ".join(f"`{kw}`" for kw in keywords['common']))
            else: st.write("_None found_")
            st.warning("**Keywords in JD (Not in Resume)**")
            if keywords['unique_text2']: st.write(", ".join(f"`{kw}`" for kw in keywords['unique_text2']))
            else: st.write("_None found_")
        with kw_col2:
            st.success("**Keywords in Resume (Maybe not in JD)**")
            if keywords['unique_text1']: st.write(", ".join(f"`{kw}`" for kw in keywords['unique_text1']))
            else: st.write("_None found_")
        st.caption("Keywords based on common Nouns & Proper Nouns (lemmatized, lowercased).")


elif st.session_state.error_message and not st.session_state.analysis_complete:
     st.error(f"Analysis could not be completed: {st.session_state.error_message}")


# --- Footer ---
st.markdown("---")
st.caption("Built by [Your Name/Team] | Using Streamlit, spaCy")