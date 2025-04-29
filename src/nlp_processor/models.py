import spacy
import streamlit as st
import os
from dotenv import load_dotenv
from typing import Optional
from spacy.language import Language
from pathlib import Path

# Import logger from helpers
from src.utils.helpers import setup_app_logger

load_dotenv()
logger = setup_app_logger(__name__)

DEFAULT_SPACY_MODEL = "en_core_web_md"
SPACY_MODEL_NAME = os.getenv("SPACY_MODEL", DEFAULT_SPACY_MODEL)
SKILL_PATTERNS_PATH = "data/skill_patterns.jsonl" # Define path here

@st.cache_resource(show_spinner="Loading NLP model...")
def load_spacy_model(model_name: str = SPACY_MODEL_NAME) -> Optional[Language]:
    """Loads a spaCy language model and optionally adds Entity Ruler for skills."""
    nlp = None # Initialize nlp as None
    try:
        logger.info(f"Attempting to load spaCy model: '{model_name}'")
        nlp = spacy.load(model_name)
        logger.info(f"Successfully loaded spaCy model: '{model_name}'")

        # Add Entity Ruler for skills IF patterns file exists
        patterns_path = Path(SKILL_PATTERNS_PATH)
        if patterns_path.is_file():
            if "entity_ruler" not in nlp.pipe_names:
                 logger.info(f"Adding EntityRuler with patterns from {patterns_path}")
                 ruler = nlp.add_pipe("entity_ruler", before="ner") # Add before built-in NER
                 try:
                      ruler.from_disk(patterns_path)
                 except Exception as e:
                      logger.error(f"Failed to load patterns from {patterns_path}: {e}. Removing ruler pipe.")
                      if "entity_ruler" in nlp.pipe_names: # Check if added before failing
                           nlp.remove_pipe("entity_ruler")
            else:
                 logger.debug("EntityRuler 'entity_ruler' already exists in pipeline.")
        else:
            logger.warning(f"Skill patterns file not found at {patterns_path}. EntityRuler not added.")

        return nlp

    except OSError:
        # ... (keep previous error handling) ...
        logger.error(f"spaCy model '{model_name}' not found. Please download it.")
        st.error(f"Model '{model_name}' not found. Run: `python -m spacy download {model_name}`")
        return None
    except Exception as e:
        # ... (keep previous error handling) ...
        logger.error(f"An unexpected error occurred loading model '{model_name}': {e}")
        st.error(f"Failed to load NLP model '{model_name}'. Please check logs.")
        # Ensure nlp is None if loading failed completely
        if nlp is not None and "entity_ruler" in nlp.pipe_names:
             try:
                  nlp.remove_pipe("entity_ruler") # Clean up partially added pipe if load failed later
             except ValueError: pass # Ignore if already removed
        return None


def get_loaded_model_name() -> str:
    """Returns the name of the spaCy model configured to be loaded."""
    return SPACY_MODEL_NAME