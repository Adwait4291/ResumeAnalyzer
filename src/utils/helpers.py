import logging
import re
import os
from typing import Optional, List
from pathlib import Path

# Optional: Only if PDF reading is needed
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None # type: ignore

# Basic Logging Setup
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_app_logger(name: str = 'AppLogger', level: str = log_level) -> logging.Logger:
    """Initializes and returns a named logger."""
    new_logger = logging.getLogger(name)
    new_logger.setLevel(level.upper())
    # Avoid adding multiple handlers if called multiple times
    if not new_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        new_logger.addHandler(handler)
    return new_logger

def clean_text(text: Optional[str]) -> str:
    """
    Cleans input text by removing excessive whitespace and normalizing line breaks.

    Args:
        text: The input string or None.

    Returns:
        The cleaned string, or an empty string if input is None or invalid.
    """
    if not isinstance(text, str):
        logger.warning("clean_text received non-string input, returning empty string.")
        return ""

    # Replace multiple newlines/spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def read_text_file(file_path: Path) -> Optional[str]:
    """Reads content from a text file."""
    try:
        logger.info(f"Reading text file: {file_path}")
        return file_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except UnicodeDecodeError:
        logger.warning(f"Could not decode file {file_path} as UTF-8, trying latin-1")
        try:
            return file_path.read_text(encoding='latin-1')
        except Exception as e:
            logger.error(f"Failed to read file {file_path} with latin-1: {e}")
            return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}")
        return None

def read_pdf_file(file_path: Path) -> Optional[str]:
    """Reads text content from a PDF file using PyPDF2 (if installed)."""
    if PyPDF2 is None:
        logger.error("PyPDF2 is not installed. Cannot read PDF files. Install with 'pip install PyPDF2'")
        return None

    text_content = []
    try:
        logger.info(f"Reading PDF file: {file_path}")
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(reader.pages)
            logger.debug(f"PDF has {num_pages} pages.")
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text_content.append(page.extract_text())
        logger.info(f"Successfully extracted text from PDF: {file_path}")
        return "\n".join(filter(None, text_content)) # Join non-empty page texts
    except FileNotFoundError:
        logger.error(f"PDF File not found: {file_path}")
        return None
    except PyPDF2.errors.PdfReadError as e:
         logger.error(f"Error reading PDF file {file_path} (possibly corrupted or password-protected): {e}")
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred reading PDF {file_path}: {e}")
        return None

def get_common_keywords(doc1, doc2, pos_tags: List[str] = ['NOUN', 'PROPN']) -> List[str]:
    """
    Extracts common keywords (lemmatized, lowercased) from two spaCy Docs
    based on specified Part-of-Speech tags, excluding stop words.

    Args:
        doc1: The first spaCy Doc object.
        doc2: The second spaCy Doc object.
        pos_tags: A list of POS tags to consider for keywords (e.g., ['NOUN', 'PROPN']).

    Returns:
        A list of common keywords found in both documents.
    """
    try:
        # Extract lemmas for specified POS tags, excluding stop words and punctuation
        keywords1 = {
            token.lemma_.lower() for token in doc1
            if token.pos_ in pos_tags and not token.is_stop and not token.is_punct
        }
        keywords2 = {
            token.lemma_.lower() for token in doc2
            if token.pos_ in pos_tags and not token.is_stop and not token.is_punct
        }

        common = list(keywords1.intersection(keywords2))
        logger.debug(f"Found {len(common)} common keywords: {common[:10]}...") # Log first few
        return sorted(common)
    except Exception as e:
        logger.error(f"Error extracting common keywords: {e}")
        return []