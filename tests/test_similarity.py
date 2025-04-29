import pytest
from pathlib import Path
import io

# Import functions from helpers
from src.utils.helpers import (
    clean_text,
    get_readability_scores,
    check_contact_info,
    check_section_headings,
    find_quantifiable_achievements,
    # Add imports for other testable functions like skill extraction if feasible
)
# Import readability if needed for direct comparison
try:
    import textstat
    readability_available_test = True
except ImportError:
    readability_available_test = False


# --- Tests for clean_text ---
@pytest.mark.parametrize("input_text, expected_output", [
    ("  hello \n world  ", "hello world"),
    ("line1\n\n\nline2", "line1\n\nline2"),
    (None, ""),
    ("\t tabbed \t ", "tabbed"),
])
def test_clean_text(input_text, expected_output):
    assert clean_text(input_text) == expected_output

# --- Tests for readability (skip if library not installed) ---
@pytest.mark.skipif(not readability_available_test, reason="textstat not installed")
def test_readability_scores_valid():
    # Generate long text
    sample_text = ("This is a sample sentence repeated many times to ensure sufficient length for calculation. " * 10 +
                   "It contains various words and structures typical of standard English prose, facilitating readability analysis. " * 5)
    scores = get_readability_scores(sample_text)
    assert scores is not None
    assert "flesch_reading_ease" in scores
    assert "flesch_kincaid_grade" in scores
    assert scores["flesch_reading_ease"] > 0 # Basic sanity check
    assert scores["flesch_kincaid_grade"] > 0

@pytest.mark.skipif(not readability_available_test, reason="textstat not installed")
def test_readability_scores_short_text():
     short_text = "This text is too short."
     assert get_readability_scores(short_text) is None

# --- Tests for ATS checks ---
def test_contact_info():
    text_with_both = "My contact: test@example.com and (123) 456-7890."
    text_with_email = "Email me at test@example.com please."
    text_with_phone = "Call 123-456-7890 for info."
    text_with_none = "No contact details here."
    assert check_contact_info(text_with_both) == {"email_found": True, "phone_found": True}
    assert check_contact_info(text_with_email) == {"email_found": True, "phone_found": False}
    assert check_contact_info(text_with_phone) == {"email_found": False, "phone_found": True}
    assert check_contact_info(text_with_none) == {"email_found": False, "phone_found": False}

def test_section_headings():
    text = """
SUMMARY
A great developer.

EDUCATION
Degree U

Work History
Company X

SKILLS & COMPETENCIES
Python, Java
"""
    expected = {"summary": True, "education": True, "experience": True, "skills": True}
    assert check_section_headings(text) == expected

# --- Tests for Achievements ---
def test_quantifiable_achievements():
    text = """
- Increased sales by 20%.
* Managed a budget of $50k.
• Reduced errors significantly. Led a team of 5.
Regular sentence.
- Saved the company time.
Achieved target goals.
    """
    results = find_quantifiable_achievements(text)
    assert len(results) >= 4 # Expecting lines with %, $, numbers, or strong verbs with bullets
    assert any("20%" in r for r in results)
    assert any("$50k" in r for r in results)
    assert any("team of 5" in r for r in results)
    # "Reduced errors" might be missed if it doesn't fit the simplified regex perfectly
    # "Saved the company time" might be missed if not bulleted AND no number

# Add more tests for skill extraction (might require mocking nlp), action verbs etc.
# Testing file readers requires creating mock file streams (io.BytesIO)