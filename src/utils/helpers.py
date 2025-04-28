import logging
import re
import os
from typing import Optional, List, IO # IO is needed for stream type hinting
from pathlib import Path
import io # Needed for BytesIO if testing streams

# Optional: Only if PDF reading is needed
try:
    import PyPDF2
    logger.info("PyPDF2 found.")
except ImportError:
    PyPDF2 = None # type: ignore
    logger.warning("PyPDF2 not found. PDF processing will be disabled.")

# Optional: Only if DOCX reading is needed
try:
    import docx # from python-docx
    logger.info("python-docx found.")
except ImportError:
    docx = None # type: ignore
    logger.warning("python-docx not found. DOCX processing will be disabled.")


# Basic Logging Setup (keep as before)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# setup_app_logger function (keep as before)
def setup_app_logger(name: str = 'AppLogger', level: str = log_level) -> logging.Logger:
    """Initializes and returns a named logger."""
    new_logger = logging.getLogger(name)
    new_logger.setLevel(level.upper())
    if not new_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        new_logger.addHandler(handler)
    return new_logger

# clean_text function (keep as before)
def clean_text(text: Optional[str]) -> str:
    """Cleans input text."""
    if not isinstance(text, str):
        logger.warning("clean_text received non-string input, returning empty string.")
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# --- File Readers from Streams (for Streamlit Uploads) ---

def read_text_from_stream(file_stream: IO[bytes], filename: str) -> Optional[str]:
    """Reads text from a bytes stream, trying UTF-8 then latin-1."""
    logger.info(f"Reading text stream: {filename}")
    try:
        # Reset buffer position (important for Streamlit uploads)
        file_stream.seek(0)
        return file_stream.read().decode("utf-8")
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decoding failed for {filename}. Trying latin-1.")
        try:
            file_stream.seek(0)
            return file_stream.read().decode("latin-1")
        except Exception as e_latin1:
            logger.error(f"Failed to read {filename} with any encoding: {e_latin1}")
            return None
    except Exception as e:
        logger.error(f"An unexpected error occurred reading text stream {filename}: {e}")
        return None

def read_pdf_from_stream(file_stream: IO[bytes], filename: str) -> Optional[str]:
    """Reads text content from a PDF bytes stream using PyPDF2 (if installed)."""
    if PyPDF2 is None:
        logger.error("PyPDF2 is not installed. Cannot read PDF stream.")
        return None

    text_content = []
    try:
        logger.info(f"Reading PDF stream: {filename}")
        # Reset buffer position
        file_stream.seek(0)
        reader = PyPDF2.PdfReader(file_stream) # PyPDF2 can read from file-like objects
        num_pages = len(reader.pages)
        logger.debug(f"PDF stream {filename} has {num_pages} pages.")
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            extracted = page.extract_text()
            if extracted: # Avoid adding None if extraction fails for a page
                 text_content.append(extracted)
        logger.info(f"Successfully extracted text from PDF stream: {filename}")
        return "\n".join(text_content)
    except PyPDF2.errors.PdfReadError as e:
         logger.error(f"Error reading PDF stream {filename} (possibly corrupted or password-protected): {e}")
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred reading PDF stream {filename}: {e}")
        return None

def read_docx_from_stream(file_stream: IO[bytes], filename: str) -> Optional[str]:
    """Reads text content from a DOCX bytes stream using python-docx (if installed)."""
    if docx is None:
        logger.error("python-docx is not installed. Cannot read DOCX stream.")
        return None

    try:
        logger.info(f"Reading DOCX stream: {filename}")
        # Reset buffer position
        file_stream.seek(0)
        document = docx.Document(file_stream) # python-docx can read from file-like objects
        text_content = [p.text for p in document.paragraphs if p.text]
        logger.info(f"Successfully extracted text from DOCX stream: {filename}")
        return "\n".join(text_content)
    except Exception as e:
        # python-docx can raise various errors (e.g., PackageNotFoundError, zipfile errors)
        logger.error(f"An unexpected error occurred reading DOCX stream {filename}: {e}")
        return None

# --- Keep Path-based readers if needed elsewhere, otherwise they can be removed ---
# def read_text_file(file_path: Path) -> Optional[str]: ...
# def read_pdf_file(file_path: Path) -> Optional[str]: ...
# def read_docx_file(file_path: Path) -> Optional[str]: ... # Would need to be added if path reading needed

# get_common_keywords function (keep as before)
def get_common_keywords(doc1, doc2, pos_tags: List[str] = ['NOUN', 'PROPN']) -> List[str]:
    # ... (implementation remains the same) ...
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