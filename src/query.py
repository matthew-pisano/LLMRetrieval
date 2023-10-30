import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


class Query:

    def __init__(self, query_id: int, query_text: str):
        self.query_id = query_id
        self.query_text = query_text

    def __repr__(self):
        return f"Query({self.query_id})"


class Doc:

    def __init__(self, docno: str, doctext: str, score: float):
        self.docno = docno
        self.doctext = doctext
        self.score = score

    def stripped(self):
        return re.sub(" "+" | ".join(stop_words)+" ", " ", self.doctext)

    def __repr__(self):
        return f"Doc({self.docno}, {self.score})"


class QueryResult:

    def __init__(self, query: Query, result_docs: list[Doc]):
        self.query = query
        self.result_docs = result_docs

    def __getitem__(self, item):
        return self.result_docs[item]

    def __repr__(self):
        return f"QueryResult({self.query.query_id}, len={len(self.result_docs)})"
