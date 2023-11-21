import re

import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def strip_stopwords(text: str):
    """Strips a string of stopwords and newlines while making it lowercase

    Args:
        text: The string to strip
    Returns:
        The stripped and formatted string"""

    return re.sub(" " + " | ".join(stop_words) + " ", " ", text).replace("\n", " ").lower()
