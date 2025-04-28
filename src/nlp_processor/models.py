import spacy
import streamlit as st
import os
from dotenv import load_dotenv
from typing import Optional
from spacy.language import Language

# Import logger from helpers
from src.utils.helpers import setup_app_logger

# Load environment variables from .env file (if it exists)
load_dotenv()

logger = setup_app_logger(__name__)

# Determine spaCy model from environment variable or use default
DEFAULT_SPACY_MODEL = "en_core_web_md" # Use _md for better similarity
SPACY_MODEL_NAME = os.getenv("SPACY_MODEL", DEFAULT_SPACY_MODEL)

@st.cache_resource(show_spinner="Loading NLP model...") # Add spinner message
def load_spacy_model(model_name: str = SPACY_MODEL_NAME) -> Optional[Language]:
    """
    Loads a spaCy language model, utilizing Streamlit's caching.

    Args:
        model_name: The name of the spaCy model to load. Defaults to the
                    value from the SPACY_MODEL environment variable or
                    'en_core_web_md'.

    Returns:
        The loaded spaCy language model object, or None if loading fails.
    """
    try:
        logger.info(f"Attempting to load spaCy model: '{model_name}'")
        # Disable unnecessary pipes if only using for similarity/basic tagging
        # This can speed up loading and processing. Adjust as needed.
        # disable_pipes = ["parser", "ner"] if model_name.startswith("en_core_web") else []
        # nlp = spacy.load(model_name, disable=disable_pipes)
        nlp = spacy.load(model_name)
        logger.info(f"Successfully loaded spaCy model: '{model_name}'")
        return nlp
    except OSError:
        logger.error(
            f"spaCy model '{model_name}' not found. "
            f"Please download it by running: \n"
            f"python -m spacy download {model_name}"
        )
        st.error(
            f"spaCy model '{model_name}' not found. Please ensure it's downloaded. "
            f"Run in your terminal: `python -m spacy download {model_name}`"
        )
        return None # Return None instead of stopping the app immediately
    except Exception as e:
        logger.error(f"An unexpected error occurred loading model '{model_name}': {e}")
        st.error(f"Failed to load NLP model '{model_name}'. Please check logs.")
        return None

# --- Example: Get model name being used ---
def get_loaded_model_name() -> str:
    """Returns the name of the spaCy model configured to be loaded."""
    return SPACY_MODEL_NAME