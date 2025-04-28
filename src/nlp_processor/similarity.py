import spacy
import streamlit as st
from typing import Optional
from spacy.tokens import Doc
from spacy.language import Language

# Import logger from helpers
from src.utils.helpers import setup_app_logger, get_common_keywords

logger = setup_app_logger(__name__)

def calculate_similarity(nlp_model: Optional[Language], text1: str, text2: str) -> float:
    """
    Calculates the semantic similarity between two texts using a loaded spaCy model.

    Args:
        nlp_model: The loaded spaCy language model. Can be None.
        text1: The first text (e.g., resume).
        text2: The second text (e.g., job description).

    Returns:
        A similarity score between 0.0 and 1.0. Returns 0.0 if inputs are
        invalid, model is missing, or vectors are unavailable.
    """
    if nlp_model is None:
        logger.error("calculate_similarity called with no spaCy model loaded.")
        st.error("NLP model is not available. Cannot calculate similarity.")
        return 0.0

    if not text1 or not text2:
        logger.warning("calculate_similarity called with empty text input(s).")
        # No need for st.warning here as app.py should handle this
        return 0.0

    try:
        logger.debug("Processing text1 and text2 with spaCy model...")
        doc1: Doc = nlp_model(text1)
        doc2: Doc = nlp_model(text2)
        logger.debug("Text processing complete.")

        # Check for vectors - essential for reliable similarity
        if not doc1.has_vector or not doc2.has_vector:
            model_name = nlp_model.meta.get("name", "Unknown")
            logger.warning(
                f"Model '{model_name}' lacks word vectors. Similarity score "
                "might be unreliable or based on context-sensitive tensors."
            )
            st.warning(
                f"The loaded model ('{model_name}') doesn't have full word vectors. "
                "Similarity results might be less accurate. Consider using 'en_core_web_md' "
                "or 'en_core_web_lg'."
            )
            # Depending on spaCy version and model, similarity might still return something
            # but it's safer to return 0 or handle appropriately if vectors are mandatory.
            # Let's proceed but be aware. You could return 0.0 here.

        # Perform similarity calculation
        logger.debug("Calculating similarity score...")
        similarity_score = doc1.similarity(doc2)
        logger.info(f"Calculated similarity score: {similarity_score:.4f}")

        # Clamp score to [0, 1] range due to potential floating point inaccuracies
        clamped_score = max(0.0, min(1.0, float(similarity_score)))
        if clamped_score != similarity_score:
             logger.debug(f"Clamped similarity score from {similarity_score} to {clamped_score}")

        return clamped_score

    except Exception as e:
        logger.error(f"Error during similarity calculation: {e}", exc_info=True)
        st.error("An unexpected error occurred during analysis.")
        return 0.0

def extract_and_compare_keywords(nlp_model: Optional[Language], text1: str, text2: str) -> dict:
    """
    Extracts and compares keywords (NOUNs, PROPNs) between two texts.

    Args:
        nlp_model: The loaded spaCy language model.
        text1: The first text.
        text2: The second text.

    Returns:
        A dictionary containing lists of keywords unique to each text and
        those common to both. Returns empty lists if error occurs.
    """
    results = {"common": [], "unique_text1": [], "unique_text2": []}
    if nlp_model is None or not text1 or not text2:
        logger.warning("Keyword extraction skipped due to missing model or text.")
        return results

    try:
        doc1 = nlp_model(text1)
        doc2 = nlp_model(text2)

        # Define POS tags for keyword extraction
        pos_tags_to_extract = ['NOUN', 'PROPN']

        # Extract keywords (lemmatized, lowercased, no stop words/punctuation)
        keywords1 = {
            token.lemma_.lower() for token in doc1
            if token.pos_ in pos_tags_to_extract and not token.is_stop and not token.is_punct and len(token.lemma_) > 1
        }
        keywords2 = {
            token.lemma_.lower() for token in doc2
            if token.pos_ in pos_tags_to_extract and not token.is_stop and not token.is_punct and len(token.lemma_) > 1
        }

        results["common"] = sorted(list(keywords1.intersection(keywords2)))
        results["unique_text1"] = sorted(list(keywords1.difference(keywords2)))
        results["unique_text2"] = sorted(list(keywords2.difference(keywords1)))

        logger.info(f"Keyword analysis complete. Common: {len(results['common'])}, Unique T1: {len(results['unique_text1'])}, Unique T2: {len(results['unique_text2'])}")
        return results

    except Exception as e:
        logger.error(f"Error during keyword extraction: {e}", exc_info=True)
        st.error("An error occurred during keyword analysis.")
        return results # Return empty dict on error