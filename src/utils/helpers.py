import logging
import re
import os
from typing import Optional, List, IO, Dict, Any, Tuple, Set
from pathlib import Path
import io
from spacy.language import Language
from spacy.tokens import Doc

# --- Library Imports with Error Handling ---
# spaCy loaded/checked in models.py

try:
    import PyPDF2
    pdf_available = True
except ImportError:
    PyPDF2 = None
    pdf_available = False

try:
    import docx
    docx_available = True
except ImportError:
    docx = None
    docx_available = False

try:
    import textstat
    readability_available = True
except ImportError:
    textstat = None
    readability_available = False

try:
    import requests
    from bs4 import BeautifulSoup
    scraping_available = True
except ImportError:
    requests = None
    BeautifulSoup = None
    scraping_available = False

# --- Basic Logging Setup ---
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Log library availability status
logger.info(f"PyPDF2 available: {pdf_available}")
logger.info(f"python-docx available: {docx_available}")
logger.info(f"textstat available: {readability_available}")
logger.info(f"requests/BeautifulSoup available: {scraping_available}")


# --- Core Helper Functions (Clean Text, Stream Readers - updated logging) ---

def clean_text(text: Optional[str]) -> str:
    """Cleans input text."""
    if not isinstance(text, str):
        logger.debug("clean_text received non-string input.")
        return ""
    # Consolidate whitespace but keep paragraph breaks (double newlines) for readability checks
    text = re.sub(r'(\r\n|\r|\n){3,}', '\n\n', text) # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text) # Consolidate spaces/tabs
    text = text.strip()
    return text

def read_text_from_stream(file_stream: IO[bytes], filename: str) -> Optional[str]:
    """Reads text from a bytes stream, trying UTF-8 then latin-1."""
    logger.info(f"Reading text stream: {filename}")
    encodings = ['utf-8', 'latin-1', 'cp1252'] # Add more if needed
    for enc in encodings:
        try:
            file_stream.seek(0)
            return file_stream.read().decode(enc)
        except (UnicodeDecodeError, TypeError):
            logger.debug(f"Decoding {filename} with {enc} failed.")
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred reading text stream {filename} with {enc}: {e}")
            return None # Stop on unexpected error
    logger.error(f"Could not decode {filename} with any attempted encoding.")
    return None

def read_pdf_from_stream(file_stream: IO[bytes], filename: str) -> Optional[str]:
    """Reads text content from a PDF bytes stream using PyPDF2 (if installed)."""
    if not pdf_available:
        logger.error("PyPDF2 is not available. Cannot read PDF stream.")
        return None
    # ... (rest of implementation from previous answer, ensure logger is used) ...
    text_content = []
    try:
        logger.info(f"Reading PDF stream: {filename}")
        file_stream.seek(0)
        reader = PyPDF2.PdfReader(file_stream)
        num_pages = len(reader.pages)
        logger.debug(f"PDF stream {filename} has {num_pages} pages.")
        for page_num in range(num_pages):
            try:
                 page = reader.pages[page_num]
                 extracted = page.extract_text()
                 if extracted:
                      text_content.append(extracted)
            except Exception as page_e:
                 logger.warning(f"Could not extract text from page {page_num+1} of {filename}: {page_e}")
        logger.info(f"Successfully extracted text from PDF stream: {filename}")
        return "\n".join(text_content)
    # ... (error handling as before) ...
    except PyPDF2.errors.PdfReadError as e:
         logger.error(f"Error reading PDF stream {filename} (possibly corrupted or password-protected): {e}")
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred reading PDF stream {filename}: {e}")
        return None

def read_docx_from_stream(file_stream: IO[bytes], filename: str) -> Optional[str]:
    """Reads text content from a DOCX bytes stream using python-docx (if installed)."""
    if not docx_available:
        logger.error("python-docx is not available. Cannot read DOCX stream.")
        return None
    # ... (rest of implementation from previous answer, ensure logger is used) ...
    try:
        logger.info(f"Reading DOCX stream: {filename}")
        file_stream.seek(0)
        document = docx.Document(file_stream)
        text_content = [p.text for p in document.paragraphs if p.text]
        logger.info(f"Successfully extracted text from DOCX stream: {filename}")
        return "\n".join(text_content)
    # ... (error handling as before) ...
    except Exception as e:
        logger.error(f"An unexpected error occurred reading DOCX stream {filename}: {e}")
        return None

# --- New Analysis Helper Functions ---

def extract_skills(nlp: Language, text: str, patterns_path: Optional[str] = "data/skill_patterns.jsonl") -> List[str]:
    """Extracts skills using spaCy's EntityRuler based on patterns."""
    if patterns_path and Path(patterns_path).is_file():
        if "entity_ruler" not in nlp.pipe_names:
            logger.info(f"Adding EntityRuler from {patterns_path}")
            ruler = nlp.add_pipe("entity_ruler", before="ner") # Add before NER if NER exists
            try:
                 ruler.from_disk(patterns_path)
            except Exception as e:
                 logger.error(f"Failed to load patterns from {patterns_path}: {e}. Skill extraction might be limited.")
                 # Remove the broken pipe
                 nlp.remove_pipe("entity_ruler")
        # else: logger.debug("EntityRuler already exists in pipeline.") # Avoid re-adding
    else:
        logger.warning(f"Skill patterns file not found at {patterns_path}. Cannot perform pattern-based skill extraction.")
        # Fallback or simply return empty if no patterns
        # return [] # Option 1: return empty if no patterns

    # Process text and extract entities labeled as "SKILL"
    doc = nlp(text)
    skills = sorted(list(set(ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL")))
    logger.info(f"Extracted {len(skills)} unique skills.")
    logger.debug(f"Skills found: {skills}")
    return skills

def check_contact_info(text: str) -> Dict[str, bool]:
    """Performs basic checks for presence of email and phone number."""
    results = {"email_found": False, "phone_found": False}
    # Simple regex (can be improved for robustness)
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    # Basic phone regex (catches common North American, some international formats - needs improvement for global use)
    phone_regex = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'

    if re.search(email_regex, text):
        results["email_found"] = True
        logger.debug("Email pattern found in text.")
    else:
         logger.debug("Email pattern not found.")

    if re.search(phone_regex, text):
        results["phone_found"] = True
        logger.debug("Phone pattern found in text.")
    else:
         logger.debug("Phone pattern not found.")

    return results

def check_section_headings(text: str) -> Dict[str, bool]:
    """Checks for presence of common section headings (case-insensitive)."""
    headings = {
        "experience": r'(professional\s+)?experience|work\s+history|employment',
        "education": r'education|academic\s+background',
        "skills": r'skills|technical\s+skills|competencies',
        "summary": r'summary|objective|profile'
    }
    results = {key: False for key in headings}
    lines = text.split('\n')

    for line in lines:
        cleaned_line = line.strip()
        if len(cleaned_line) > 0 and len(cleaned_line) < 50: # Avoid checking long paragraphs
            for key, pattern in headings.items():
                if re.search(pattern, cleaned_line, re.IGNORECASE):
                    results[key] = True
                    logger.debug(f"Found potential heading for '{key}': {cleaned_line}")
                    # Optional: break if one match per line is enough
                    # break

    logger.info(f"Heading check results: {results}")
    return results

def find_quantifiable_achievements(text: str) -> List[str]:
    """Attempts to find sentences suggesting quantifiable results using regex."""
    # Regex looking for numbers, %, $, or impact words near numbers/start of line (bullet points)
    # This is a basic heuristic and may have false positives/negatives.
    pattern = r'^[*\-•\s]*.*?(?:[\d.,$%]+|\b(?:increased|decreased|reduced|managed|led|achieved|saved|grew|improved)\b).*$'
    achievements = []
    # Split into potential bullet points or sentences
    potential_lines = re.split(r'\n|\.', text) # Split by newline or period

    for line in potential_lines:
        line = line.strip()
        if line and re.match(pattern, line, re.IGNORECASE):
            # Further check: ensure it contains a number or a strong verb if no number?
            has_number = bool(re.search(r'\d', line))
            has_strong_verb = bool(re.search(r'\b(increased|decreased|reduced|managed|led|achieved|saved|grew|improved)\b', line, re.IGNORECASE))
            # Require either a number or start with a bullet-like char AND a strong verb
            is_bullet_start = bool(re.match(r'^[*\-•\s]+', line))

            if has_number or (is_bullet_start and has_strong_verb):
                 # Limit length to avoid overly long matches
                achievements.append(line[:250] + "..." if len(line) > 250 else line)

    logger.info(f"Found {len(achievements)} potential quantifiable achievement lines.")
    logger.debug(f"Potential achievements: {achievements}")
    return achievements

def extract_action_verbs(nlp: Language, text: str) -> List[str]:
    """Extracts common action verbs (past tense) potentially from bullet points."""
    verbs = []
    doc = nlp(text)
    potential_bullet_starts = ('*', '-', '•') # Common bullet point characters

    for sent in doc.sents:
        sent_text = sent.text.strip()
        # Check if sentence starts like a bullet point and the first token is a past tense verb
        if sent_text and sent_text.startswith(potential_bullet_starts):
            first_token = sent[1] if sent[0].is_space else sent[0] # Handle potential leading space after bullet
            if first_token.pos_ == 'VERB' and first_token.tag_ == 'VBD': # VBD is past tense verb tag in Penn Treebank
                verbs.append(first_token.lemma_.lower())

    # Count frequency and return most common (e.g., top 10)
    from collections import Counter
    verb_counts = Counter(verbs)
    common_verbs = [verb for verb, count in verb_counts.most_common(15)]
    logger.info(f"Extracted {len(common_verbs)} common potential action verbs (past tense) from bullet points.")
    logger.debug(f"Common action verbs: {common_verbs}")
    return common_verbs


def get_readability_scores(text: str) -> Optional[Dict[str, float]]:
    """Calculates readability scores using textstat (if available)."""
    if not readability_available:
        logger.error("textstat library not available. Cannot calculate readability.")
        return None
    if not text or len(text.split()) < 100: # textstat requires minimum words
         logger.warning("Text too short (<100 words) for reliable readability scores.")
         return None
    try:
        scores = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            # Add more scores if desired (e.g., textstat.gunning_fog(text))
        }
        logger.info(f"Calculated readability scores: {scores}")
        return scores
    except Exception as e:
        logger.error(f"Error calculating readability scores: {e}")
        return None

def scrape_job_description(url: str) -> Optional[str]:
    """Attempts to scrape job description text from a URL."""
    if not scraping_available:
        logger.error("requests or beautifulsoup4 not found. Cannot scrape URL.")
        return None
    if not url or not url.startswith(('http://', 'https://')):
        logger.warning(f"Invalid URL provided for scraping: {url}")
        return None

    try:
        logger.info(f"Attempting to scrape job description from: {url}")
        headers = { # Add headers to mimic a browser
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10) # Add timeout
        response.raise_for_status() # Raise error for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'lxml') # Use lxml parser

        # --- Very Basic Scraping Logic (Needs Improvement for Specific Sites) ---
        # Try to find common tags/classes used for job descriptions. This is highly site-specific.
        possible_containers = soup.find_all(['div', 'article', 'section'],
                                           attrs={'class': [re.compile(r'job.?description', re.I),
                                                            re.compile(r'job.?details', re.I),
                                                            re.compile(r'description.?content', re.I)]})
        text_parts = []
        if possible_containers:
            logger.debug(f"Found {len(possible_containers)} potential description containers.")
            container = possible_containers[0] # Assume first is best for simplicity
            # Extract text from paragraphs, list items, etc. within the container
            for element in container.find_all(['p', 'li', 'div'], recursive=True):
                 # Avoid script/style tags and overly short/long text snippets
                 if element.name not in ['script', 'style'] and len(element.get_text(strip=True)) > 10:
                      text_parts.append(element.get_text(separator=' ', strip=True))
        else:
            # Fallback: Get text from the main body, hoping it's mostly the JD
            logger.warning("Could not find specific job description container, attempting to parse main body.")
            body = soup.find('body')
            if body:
                 text_parts = [p.get_text(separator=' ', strip=True) for p in body.find_all('p') if len(p.get_text(strip=True)) > 20]

        if not text_parts:
             logger.warning(f"Could not extract significant text content from {url}")
             return None

        full_text = "\n".join(text_parts)
        cleaned_text = clean_text(full_text) # Clean the extracted text
        logger.info(f"Successfully scraped and cleaned text from {url} (Length: {len(cleaned_text)})")
        return cleaned_text

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Error during scraping {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing or processing scraped content from {url}: {e}")
        return None

# --- Report Generation ---
def generate_report_text(analysis_results: Dict[str, Any]) -> str:
    """Generates a plain text report from the analysis results dictionary."""
    report = []
    report.append("="*30)
    report.append("   Resume Analysis Report")
    report.append("="*30 + "\n")

    # 1. Overall Score
    score = analysis_results.get('similarity_score', 0.0)
    report.append(f"Overall Alignment Score: {score:.1%}")
    interpretation = "Needs Improvement"
    if score > 0.90: interpretation = "Excellent!"
    elif score > 0.80: interpretation = "Good"
    elif score > 0.70: interpretation = "Moderate"
    report.append(f"Interpretation: {interpretation}\n")

    # 2. Keyword Analysis
    kw = analysis_results.get('keywords', {})
    report.append("-" * 20)
    report.append("Keyword Analysis:")
    report.append("-" * 20)
    report.append(f"Common Keywords: {', '.join(kw.get('common', [])) if kw.get('common') else 'None found'}")
    report.append(f"Keywords in JD (Not in Resume): {', '.join(kw.get('unique_jd', [])) if kw.get('unique_jd') else 'None found'}")
    report.append(f"Keywords in Resume (Not in JD): {', '.join(kw.get('unique_resume', [])) if kw.get('unique_resume') else 'None found'}\n")

    # 3. Skill Analysis (if available)
    skills = analysis_results.get('skills', {})
    if skills:
        report.append("-" * 20)
        report.append("Skill Analysis:")
        report.append("-" * 20)
        report.append(f"Skills Found in Resume: {', '.join(skills.get('resume_skills', [])) or 'None identified'}")
        report.append(f"Skills Required by Job Desc (extracted): {', '.join(skills.get('jd_skills', [])) or 'None identified'}")
        report.append(f"Matching Skills: {', '.join(skills.get('matching_skills', [])) or 'None'}")
        report.append(f"Missing Skills: {', '.join(skills.get('missing_skills', [])) or 'None'}\n")


    # 4. ATS Friendliness Checks
    ats = analysis_results.get('ats_checks', {})
    report.append("-" * 20)
    report.append("ATS Friendliness Checks:")
    report.append("-" * 20)
    report.append(f"Contact Info Found: Email ({ats.get('email_found', False)}), Phone ({ats.get('phone_found', False)})")
    report.append("Common Sections Found:")
    for section, found in ats.get('sections', {}).items():
        report.append(f"  - {section.capitalize()}: {'Yes' if found else 'No'}")
    # Add more ATS checks here (e.g., date format results if implemented)
    report.append("") # Add newline

    # 5. Quantifiable Achievements
    achievements = analysis_results.get('achievements', [])
    report.append("-" * 20)
    report.append("Potential Quantifiable Achievements:")
    report.append("-" * 20)
    if achievements:
        for i, ach in enumerate(achievements):
            report.append(f"{i+1}. {ach}")
    else:
        report.append("None explicitly identified (or feature needs refinement). Consider adding more measurable results.")
    report.append("") # Add newline

    # 6. Action Verbs
    verbs = analysis_results.get('action_verbs', [])
    report.append("-" * 20)
    report.append("Common Action Verbs (Past Tense, from bullet points):")
    report.append("-" * 20)
    report.append(f"{', '.join(verbs) or 'None identified (or feature needs refinement).'}\n")

    # 7. Readability
    readability = analysis_results.get('readability', {})
    report.append("-" * 20)
    report.append("Readability Scores (Resume):")
    report.append("-" * 20)
    if readability and readability_available:
        report.append(f"Flesch Reading Ease: {readability.get('flesch_reading_ease', 'N/A'):.2f} (Higher is easier)")
        report.append(f"Flesch-Kincaid Grade Level: {readability.get('flesch_kincaid_grade', 'N/A'):.2f}")
    elif not readability_available:
         report.append("textstat library not installed.")
    else:
         report.append("Could not calculate (text might be too short or an error occurred).")
    report.append("\n" + "="*30)

    return "\n".join(report)