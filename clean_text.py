import re
import unicodedata

def clean_text(text):
    """
    Clean LLM-generated text by:
    - Removing hidden/invisible Unicode characters
    - Stripping HTML tags (if any)
    - Eliminating 'weird' special characters (excluding . , ; : ' " ? ! -)
    - Replacing multiple punctuation (e.g., '!!!') with a single instance
    - Collapsing multiple whitespace into one space
    - Trimming leading/trailing whitespace
    """
    # Normalize unicode (e.g., accents)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")

    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Remove unwanted special characters (allow alphanumeric and . , ; : ' " ? ! - /)
    text = re.sub(r"[^A-Za-z0-9\s\.,;:'\"\?\!\-\/]", '', text)

    # Compress repeated punctuation (e.g., '!!!' → '!')
    text = re.sub(r'([.,;:\'\"\?\!\\-])\1+', r'\1', text)

    # # Collapse multiple whitespace/newlines
    # text = re.sub(r'\s+', ' ', text)

    # Trim space
    return text.strip()




