import re

import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def strip_stopwords(text: str):
    return re.sub(" " + " | ".join(stop_words) + " ", " ", text)
