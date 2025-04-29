import streamlit as st
import os
from pathlib import Path
from typing import Optional, IO, Dict, Any
from datetime import datetime

# --- Annotated Text ---
from annotated_text import annotated_text, annotation # For highlighting

# --- Import functions from source modules ---
from src.nlp_processor.models import load_spacy_model, get_loaded_model_name
from src.nlp_processor.similarity import calculate_similarity
# Import *all* necessary helpers
from src.utils.helpers import (
    setup_app_logger, clean_text,
    read_text_from_stream, read_pdf_from_stream, read_docx_from_stream,
    extract_skills, check_contact_info, check_section_headings,
    find_quantifiable_achievements, extract_action_verbs,
    get_readability_scores, scrape_job_description, generate_report_text,
    get_common_keywords # Assuming get_common_keywords is still relevant
)

# --- Setup Logger ---
logger = setup_app_logger(__name__)

# --- App Configuration ---
st.set_page_config(
    page_title="Resume Analyzer Pro", # New name?
    page_icon="🚀",
    layout="wide",
)

# --- Load NLP Model ---
MODEL_NAME = get_loaded_model_name()
# Wrap model loading in try-except at app level too
try:
    nlp = load_spacy_model()
    if nlp is None:
        # Error already shown by load_spacy_model
        st.stop() # Stop execution if model fails critically
except Exception as e:
    logger.critical(f"Failed to initialize NLP model in app.py: {e}", exc_info=True)
    st.error("A critical error occurred loading the NLP model. Application cannot start.")
    st.stop()


# --- Helper Function for Reading Uploaded File (keep as before) ---
def get_text_from_upload(uploaded_file: Optional[IO[bytes]]) -> Optional[str]:
    # ... (Implementation from previous answer using stream readers) ...
    if uploaded_file is None: return None
    file_name = uploaded_file.name; file_type = uploaded_file.type
    logger.info(f"Processing uploaded file: {file_name} (Type: {file_type})")
    try:
        if file_type == "text/plain": return read_text_from_stream(uploaded_file, file_name)
        elif file_type == "application/pdf":
            text = read_pdf_from_stream(uploaded_file, file_name)
            if text is None: st.error(f"Could not read PDF: '{file_name}'. Check library/file.")
            return text
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
             text = read_docx_from_stream(uploaded_file, file_name)
             if text is None: st.error(f"Could not read DOCX: '{file_name}'. Check library/file.")
             return text
        else:
            logger.warning(f"Unsupported file type uploaded: {file_type} for file {file_name}")
            st.warning(f"Unsupported file type '{file_type}'. Please upload .txt, .pdf, or .docx.")
            return None
    except Exception as e:
        logger.error(f"Error processing uploaded file {file_name}: {e}", exc_info=True)
        st.error(f"An error occurred reading the file: {e}")
        return None


# --- Helper function for Highlighting ---
def create_highlighted_text(text: str, keywords: List[str], color: str = "#fea", label: str = "MATCH") -> List[Any]:
    """Creates data structure for annotated_text highlighting."""
    if not keywords:
        return [text] # Return plain text if no keywords

    annotated_list = []
    # Use regex to find keywords (case-insensitive) and split text
    # Sort keywords by length descending to match longer phrases first
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    pattern = '|'.join(r'\b' + re.escape(kw) + r'\b' for kw in sorted_keywords)
    regex = re.compile(pattern, re.IGNORECASE)

    last_end = 0
    for match in regex.finditer(text):
        start, end = match.span()
        # Add text before the match
        if start > last_end:
            annotated_list.append(text[last_end:start])
        # Add the highlighted match
        annotated_list.append(annotation(text[start:end], label, background=color, color="#000"))
        last_end = end

    # Add any remaining text after the last match
    if last_end < len(text):
        annotated_list.append(text[last_end:])

    return annotated_list


# --- Initialize Session State ---
# Keep previous state variables + add new ones for analysis results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'resume_text_display' not in st.session_state:
     st.session_state.resume_text_display = "" # For highlighting
if 'jd_text_display' not in st.session_state:
     st.session_state.jd_text_display = "" # For highlighting

# --- App UI ---
st.title("🚀 Resume Analyzer Pro")
st.markdown(f"Enhanced analysis to optimize your resume. Using spaCy model: `{MODEL_NAME}`.")

# --- Inputs ---
st.markdown("---")
st.header("Inputs")
col1, col2 = st.columns(2)

# Input variables
resume_input_method = col1.radio("1. Resume Input Method:", ("Paste Text", "Upload File"), key="resume_method", horizontal=True)
# *** ADDED: JD Input Method ***
jd_input_method = col2.radio("2. Job Description Input:", ("Paste Text", "Enter URL"), key="jd_method", horizontal=True)

resume_text_area = None
uploaded_resume_file = None
jd_text_area = None
jd_url = None # New variable for URL

with col1:
    st.subheader("Your Resume")
    if resume_input_method == "Paste Text":
        resume_text_area = st.text_area("Paste resume text:", height=350, key="resume_paste", placeholder="Paste full resume text...")
    else:
        uploaded_resume_file = st.file_uploader(
            "Upload Resume File", type=['txt', 'pdf', 'docx'], key="resume_upload",
            help="Upload in .txt, .pdf, or .docx format."
        )

with col2:
    st.subheader("Job Description")
    if jd_input_method == "Paste Text":
         jd_text_area = st.text_area("Paste job description:", height=350, key="jd_paste", placeholder="Paste full job description text...")
    else:
         jd_url = st.text_input("Enter Job Description URL:", key="jd_url", placeholder="https://www.linkedin.com/jobs/view/...")


st.markdown("---")

# Analysis Trigger
analyze_button = st.button("✨ Analyze Alignment", type="primary", use_container_width=True)


# --- Analysis Logic ---
if analyze_button:
    st.session_state.analysis_complete = False
    st.session_state.analysis_results = {} # Reset results
    st.session_state.error_message = None
    st.session_state.resume_text_display = ""
    st.session_state.jd_text_display = ""
    resume_content = None
    jd_content = None
    analysis_performed = False # Flag to track if analysis step was reached

    # 1. Get Resume Content
    with st.spinner("Reading resume..."):
        if resume_input_method == "Paste Text":
            resume_content = resume_text_area
        elif uploaded_resume_file is not None:
            resume_content = get_text_from_upload(uploaded_resume_file)
        else:
             st.warning("Please paste or upload a resume.")
             st.session_state.error_message = "Resume input missing."

    # 2. Get Job Description Content
    if resume_content: # Only proceed if resume is okay
        with st.spinner("Reading job description..."):
            if jd_input_method == "Paste Text":
                jd_content = jd_text_area
                if not jd_content:
                     st.warning("Please paste the job description.")
                     st.session_state.error_message = "Job description input missing."
            elif jd_url:
                jd_content = scrape_job_description(jd_url)
                if not jd_content:
                    st.error(f"Could not scrape job description from URL: {jd_url}. Try pasting text instead.")
                    st.session_state.error_message = "Failed to scrape URL."
            else:
                 st.warning("Please paste the job description or enter a URL.")
                 st.session_state.error_message = "Job description input missing."

    # 3. Perform Analysis if inputs are valid
    if resume_content and jd_content:
        logger.info("Inputs obtained, starting full analysis...")
        analysis_performed = True
        with st.spinner("Performing analysis (this may take a minute)..."):
            try:
                # Clean texts
                cleaned_resume = clean_text(resume_content)
                cleaned_jd = clean_text(jd_content)
                st.session_state.resume_text_display = cleaned_resume # Store for display
                st.session_state.jd_text_display = cleaned_jd       # Store for display

                if not cleaned_resume or not cleaned_jd:
                    st.error("Input text became empty after cleaning. Cannot analyze.")
                    st.session_state.error_message = "Input empty after cleaning."
                else:
                    # --- Run All Analysis Functions ---
                    results: Dict[str, Any] = {}

                    # a) Similarity Score
                    results['similarity_score'] = calculate_similarity(nlp, cleaned_resume, cleaned_jd)

                    # b) Skill Extraction (using EntityRuler)
                    resume_skills = extract_skills(nlp, cleaned_resume)
                    jd_skills = extract_skills(nlp, cleaned_jd)
                    matching_skills = sorted(list(set(resume_skills).intersection(set(jd_skills))))
                    missing_skills = sorted(list(set(jd_skills).difference(set(resume_skills))))
                    results['skills'] = {
                        "resume_skills": resume_skills,
                        "jd_skills": jd_skills,
                        "matching_skills": matching_skills,
                        "missing_skills": missing_skills
                    }

                    # c) Basic Keyword Comparison (e.g., Nouns/Proper Nouns)
                    # Need spaCy docs for this part of keyword analysis
                    resume_doc = nlp(cleaned_resume)
                    jd_doc = nlp(cleaned_jd)
                    common_kws = get_common_keywords(resume_doc, jd_doc)
                    # Calculate unique keywords based on common ones
                    resume_kws_set = {token.lemma_.lower() for token in resume_doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and not token.is_punct}
                    jd_kws_set = {token.lemma_.lower() for token in jd_doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and not token.is_punct}
                    results['keywords'] = {
                         "common": common_kws,
                         "unique_resume": sorted(list(resume_kws_set.difference(jd_kws_set))),
                         "unique_jd": sorted(list(jd_kws_set.difference(resume_kws_set)))
                     }

                    # d) ATS Checks
                    results['ats_checks'] = {
                        **check_contact_info(cleaned_resume),
                        "sections": check_section_headings(cleaned_resume)
                        # Add more checks here (date format, etc.)
                    }

                    # e) Quantifiable Achievements
                    results['achievements'] = find_quantifiable_achievements(cleaned_resume)

                    # f) Action Verbs
                    results['action_verbs'] = extract_action_verbs(nlp, cleaned_resume)

                    # g) Readability
                    results['readability'] = get_readability_scores(cleaned_resume)

                    # Store results and mark complete
                    st.session_state.analysis_results = results
                    st.session_state.analysis_complete = True
                    logger.info("Analysis complete. Stored results in session state.")
                    st.success("Analysis Complete!")

            except Exception as e:
                logger.error(f"An error occurred during the main analysis pipeline: {e}", exc_info=True)
                st.error(f"An unexpected error occurred during analysis: {e}")
                st.session_state.error_message = str(e)

# --- Display Results ---
st.markdown("---")
st.header("📊 Analysis Dashboard")

if not analysis_performed and st.session_state.error_message:
     st.error(f"Could not perform analysis: {st.session_state.error_message}")
elif not st.session_state.analysis_complete and analyze_button: # Button pressed, but analysis didn't finish
     st.info("Processing... Please wait for results to appear.")
elif not st.session_state.analysis_complete:
     st.info("Enter your resume and job description, then click 'Analyze Alignment'.")

if st.session_state.analysis_complete and st.session_state.analysis_results:
    results = st.session_state.analysis_results
    report_text = generate_report_text(results) # Generate report text

    # --- Download Button ---
    st.download_button(
        label="📥 Download Full Report (.txt)",
        data=report_text,
        file_name=f"resume_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    st.markdown("---")

    # --- Display Sections ---
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        # 1. Overall Score
        st.subheader("📈 Overall Alignment")
        score = results.get('similarity_score', 0.0)
        st.metric(label="Similarity Score", value=f"{score:.1%}")
        interpretation = "Needs Improvement"; color = "red"
        if score > 0.90: interpretation, color = "Excellent!", "blue"
        elif score > 0.80: interpretation, color = "Good", "green"
        elif score > 0.70: interpretation, color = "Moderate", "orange"
        st.markdown(f"**Interpretation:** <span style='color:{color};'>{interpretation}</span>", unsafe_allow_html=True)
        st.progress(score)
        st.divider()

        # 2. Readability
        st.subheader("👓 Readability (Resume)")
        readability = results.get('readability')
        if readability:
            st.metric(label="Flesch Reading Ease", value=f"{readability.get('flesch_reading_ease', 0):.1f}")
            st.caption("Score 0-100. Higher is easier (60-70 ideal for wide audience).")
            st.metric(label="Grade Level (Flesch-Kincaid)", value=f"{readability.get('flesch_kincaid_grade', 0):.1f}")
            st.caption("US school grade level needed to understand the text.")
        else:
            st.info("Readability score not available (text too short, library missing, or error).")
        st.divider()

        # 3. ATS Friendliness
        st.subheader("🤖 ATS Friendliness (Resume)")
        ats = results.get('ats_checks', {})
        contact = f"Email: {'✅' if ats.get('email_found') else '❌'} | Phone: {'✅' if ats.get('phone_found') else '❌'}"
        st.markdown(f"**Contact Info:** {contact}")
        st.markdown("**Common Sections Found:**")
        sections = ats.get('sections', {})
        cols = st.columns(len(sections) if sections else 1)
        if sections:
             for i, (sec, found) in enumerate(sections.items()):
                  with cols[i]:
                       st.markdown(f"- {sec.capitalize()}: {'✅' if found else '⚠️'}")
        else:
             st.markdown("- _No standard sections detected._")
        st.caption("Checks for common section headings and contact patterns.")
        st.divider()

        # 4. Quantifiable Achievements
        st.subheader("🎯 Quantifiable Achievements (Resume)")
        achievements = results.get('achievements', [])
        if achievements:
            with st.expander(f"Found {len(achievements)} potential achievement lines (click to view)", expanded=False):
                for ach in achievements:
                    st.markdown(f"- _{ach}_")
        else:
            st.warning("Few quantifiable achievements detected. Consider adding measurable results using numbers, $, %, or impact verbs.")
        st.caption("Looks for lines with numbers or strong action verbs, often starting with bullets.")


    with res_col2:
        # 5. Skill Analysis
        st.subheader("🛠️ Skill Analysis")
        skills = results.get('skills', {})
        if skills:
             st.success(f"**Matching Skills ({len(skills.get('matching_skills',[]))}):**")
             if skills.get('matching_skills'):
                  st.write(f"_{', '.join(skills['matching_skills'])}_")
             else: st.write("_None_")

             st.warning(f"**Missing Skills ({len(skills.get('missing_skills',[]))}):** (Required by JD, not found in Resume)")
             if skills.get('missing_skills'):
                  st.write(f"_{', '.join(skills['missing_skills'])}_")
             else: st.write("_None_")

             with st.expander("See all extracted skills"):
                 st.markdown(f"**Resume Skills:** {', '.join(skills.get('resume_skills',[])) or 'None identified'}")
                 st.markdown(f"**Job Description Skills:** {', '.join(skills.get('jd_skills',[])) or 'None identified'}")
        else:
             st.info("Skill analysis data not available.")
        st.caption("Based on Entity Ruler patterns from `data/skill_patterns.jsonl`.")
        st.divider()

        # 6. Keyword Analysis
        st.subheader("🔑 Keyword Comparison (Nouns/Proper Nouns)")
        kw = results.get('keywords', {})
        if kw:
             st.info(f"**Common Keywords ({len(kw.get('common',[]))}):**")
             if kw.get('common'):
                  st.write(f"_{', '.join(kw['common'])}_")
             else: st.write("_None found_")

             st.warning(f"**Keywords in JD only ({len(kw.get('unique_jd',[]))}):**")
             if kw.get('unique_jd'):
                  st.write(f"_{', '.join(kw['unique_jd'])}_")
             else: st.write("_None found_")

             # Optional: Show keywords only in resume
             # st.success(f"**Keywords in Resume only ({len(kw.get('unique_resume',[]))}):**")
             # if kw.get('unique_resume'): st.write(f"_{', '.join(kw['unique_resume'])}_")
             # else: st.write("_None found_")

        else:
             st.info("Keyword analysis data not available.")
        st.divider()

        # 7. Action Verbs
        st.subheader("🏃 Action Verbs (Resume)")
        verbs = results.get('action_verbs', [])
        if verbs:
            st.write(f"Common past-tense verbs found (potentially in experience bullet points):")
            st.write(f"_{', '.join(verbs)}_")
        else:
            st.info("No common action verbs identified in expected locations (e.g., past tense verbs at start of bullet points).")


    # --- Text Highlighting Expander ---
    st.divider()
    with st.expander(" H̲i̲g̲h̲l̲i̲g̲h̲t̲e̲d̲ ̲T̲e̲x̲t̲s̲ ", expanded=False):
         st.subheader("Resume Text with Common Keywords Highlighted")
         # Use keywords identified as common between Resume and JD for highlighting
         common_keywords_to_highlight = results.get('keywords', {}).get('common', [])
         if st.session_state.resume_text_display and common_keywords_to_highlight:
              annotated_resume = create_highlighted_text(
                   st.session_state.resume_text_display,
                   common_keywords_to_highlight,
                   color="#8ef", label="COMMON"
              )
              annotated_text(*annotated_resume)
         elif st.session_state.resume_text_display:
              st.text(st.session_state.resume_text_display) # Show plain if no keywords
         else:
              st.info("Resume text not available for highlighting.")

         st.subheader("Job Description Text with Common Keywords Highlighted")
         if st.session_state.jd_text_display and common_keywords_to_highlight:
              annotated_jd = create_highlighted_text(
                   st.session_state.jd_text_display,
                   common_keywords_to_highlight,
                   color="#8ef", label="COMMON"
              )
              annotated_text(*annotated_jd)
         elif st.session_state.jd_text_display:
              st.text(st.session_state.jd_text_display) # Show plain if no keywords
         else:
              st.info("Job description text not available for highlighting.")


# --- Footer ---
st.markdown("---")
st.caption("Resume Analyzer Pro | Built with Streamlit, spaCy, and ❤️")