import pytest
import spacy
from spacy.language import Language
import os
from dotenv import load_dotenv

# Load environment variables for consistency if model name is used from there
load_dotenv()

# Import functions to test (adjust path if necessary)
from src.nlp_processor.similarity import calculate_similarity, extract_and_compare_keywords
# Assume models.py handles loading correctly, or load directly for tests

# Determine model to use for testing (use a smaller one if possible, but need vectors)
TEST_MODEL_NAME = os.getenv("SPACY_MODEL", "en_core_web_md") # Use the same logic as app

@pytest.fixture(scope="module") # Load model only once per module
def nlp_model() -> Language:
    """Pytest fixture to load the spaCy model."""
    print(f"\nLoading test model: {TEST_MODEL_NAME}")
    try:
        # You might disable pipes you don't explicitly test for speed
        # disable_pipes = ["parser", "ner"]
        # model = spacy.load(TEST_MODEL_NAME, disable=disable_pipes)
        model = spacy.load(TEST_MODEL_NAME)
        print("Test model loaded.")
        return model
    except OSError:
        pytest.skip(f"Skipping tests: spaCy model '{TEST_MODEL_NAME}' not found. Run 'python -m spacy download {TEST_MODEL_NAME}'", allow_module_level=True)

# --- Tests for calculate_similarity ---

def test_similarity_identical(nlp_model: Language):
    """Test similarity of identical non-trivial texts."""
    text = "This is a reasonably long sentence used for testing similarity calculation."
    score = calculate_similarity(nlp_model, text, text)
    assert score == pytest.approx(1.0, abs=1e-3)

def test_similarity_very_different(nlp_model: Language):
    """Test similarity of semantically very different texts."""
    text1 = "The quick brown fox jumps over the lazy dog near the river bank."
    text2 = "Operating systems manage computer hardware and software resources efficiently."
    score = calculate_similarity(nlp_model, text1, text2)
    assert score < 0.6 # Expect low similarity, threshold might need tuning based on model

def test_similarity_related(nlp_model: Language):
    """Test similarity of semantically related texts."""
    text1 = "We are looking for a Python developer with experience in web frameworks like Django or Flask."
    text2 = "Seeking a software engineer skilled in Python programming and backend development using Django."
    score = calculate_similarity(nlp_model, text1, text2)
    assert score > 0.75 # Expect reasonably high similarity

@pytest.mark.parametrize("text1, text2, expected", [
    ("", "Some text", 0.0),
    ("Some text", "", 0.0),
    ("", "", 0.0),
    ("Short", "Different short", 0.0) # Model might struggle with very short texts
])
def test_similarity_empty_or_short(nlp_model: Language, text1: str, text2: str, expected: float):
    """Test similarity with empty or very short inputs."""
    # Note: Actual score for short texts depends heavily on the model and content.
    # Here we mainly test the handling of empty strings.
    if len(text1) < 5 or len(text2) < 5: # Check if it's an empty/short case
        assert calculate_similarity(nlp_model, text1, text2) == expected
    else: # Placeholder for potential future short text tests
        pass


def test_similarity_unicode(nlp_model: Language):
    """Test similarity with unicode characters."""
    text1 = "Résumé with special characters like éàçü."
    text2 = "An application including a résumé with accents éàçü is required."
    score = calculate_similarity(nlp_model, text1, text2)
    assert score > 0.7 # Expect relatively high similarity

def test_similarity_no_model():
    """Test behavior when NLP model is None."""
    score = calculate_similarity(None, "Some text", "Other text")
    assert score == 0.0

# --- Tests for extract_and_compare_keywords ---

def test_keywords_basic(nlp_model: Language):
    """Test basic keyword extraction and comparison."""
    text1 = "The data scientist uses Python and SQL for analysis."
    text2 = "Analysis requires Python skills and knowledge of SQL databases."
    expected = {
        "common": sorted(["analysis", "python", "sql"]),
        "unique_text1": sorted(["data", "scientist"]), # 'uses' is stopword
        "unique_text2": sorted(["database", "knowledge", "skill"]) # 'requires' verb
    }
    result = extract_and_compare_keywords(nlp_model, text1, text2)
    # Sort results for consistent comparison
    for key in result: result[key] = sorted(result[key])
    assert result == expected

def test_keywords_no_common(nlp_model: Language):
    """Test keyword extraction with no common relevant words."""
    text1 = "The cat sat on the mat." # Mostly stopwords or non-nouns/propns
    text2 = "A quick brown fox jumped."
    expected = {
        "common": [],
        "unique_text1": sorted(["cat", "mat"]),
        "unique_text2": sorted(["fox"])
    }
    result = extract_and_compare_keywords(nlp_model, text1, text2)
    for key in result: result[key] = sorted(result[key])
    assert result == expected


@pytest.mark.parametrize("text1, text2", [
    ("", "Some text"),
    ("Some text", ""),
    ("", ""),
])
def test_keywords_empty_input(nlp_model: Language, text1: str, text2: str):
    """Test keyword extraction with empty inputs."""
    expected = {"common": [], "unique_text1": [], "unique_text2": []}
    result = extract_and_compare_keywords(nlp_model, text1, text2)
    assert result == expected

def test_keywords_no_model():
    """Test keyword extraction when model is None."""
    expected = {"common": [], "unique_text1": [], "unique_text2": []}
    result = extract_and_compare_keywords(None, "Some text", "Other text")
    assert result == expected