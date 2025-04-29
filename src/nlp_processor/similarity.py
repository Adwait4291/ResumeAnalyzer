import spacy
import streamlit as st
from typing import Optional
from spacy.tokens import Doc
from spacy.language import Language

# Import logger
from src.utils.helpers import setup_app_logger

logger = setup_app_logger(__name__)

# calculate_similarity function remains largely the same as the robust version
# Ensure it uses the module logger and handles None model robustly.
def calculate_similarity(nlp_model: Optional[Language], text1: str, text2: str) -> float:
    """
    Calculates the semantic similarity between two texts using a loaded spaCy model.
    (Implementation from previous robust version)
    """
    if nlp_model is None:
        logger.error("calculate_similarity called with no spaCy model loaded.")
        # st.error("NLP model is not available...") # Let app.py handle UI errors
        return 0.0

    if not text1 or not text2:
        logger.warning("calculate_similarity called with empty text input(s).")
        return 0.0

    try:
        logger.debug("Processing text1 and text2 with spaCy model for similarity...")
        doc1: Doc = nlp_model(text1)
        doc2: Doc = nlp_model(text2)
        logger.debug("Text processing complete.")

        if not doc1.has_vector or not doc2.has_vector:
            model_name = nlp_model.meta.get("name", "Unknown")
            logger.warning(f"Model '{model_name}' lacks vectors. Similarity may be less reliable.")
            # st.warning(...) # Let app.py handle UI warnings

        logger.debug("Calculating similarity score...")
        similarity_score = doc1.similarity(doc2)
        logger.info(f"Calculated similarity score: {similarity_score:.4f}")

        clamped_score = max(0.0, min(1.0, float(similarity_score)))
        if clamped_score != similarity_score:
             logger.debug(f"Clamped similarity score from {similarity_score} to {clamped_score}")
        return clamped_score

    except Exception as e:
        logger.error(f"Error during similarity calculation: {e}", exc_info=True)
        # st.error("An unexpected error occurred during analysis.") # Let app.py handle UI
        return 0.0

# Keyword extraction is now primarily handled by helpers.py/app.py
# extract_and_compare_keywords function can be removed from here if desired,
# or kept if used internally for other similarity metrics later.