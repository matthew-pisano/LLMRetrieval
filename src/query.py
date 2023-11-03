import math
import re


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

    def __len__(self):
        return len(self.doctext)

    def __repr__(self):
        return f"Doc({self.docno}, {self.score})"


class QueryResult:

    def __init__(self, query: Query, result_docs: list[Doc]):
        self.query = query
        self.docs = result_docs
        self.scores = [doc.score for doc in self.docs]
        self.max_score = max(self.scores)
        self.cg = sum(self.scores)
        self.dcg = sum([score/math.log2(i+2) for i, score in enumerate(self.scores)])

    def __getitem__(self, item):
        return self.docs[item]

    def __len__(self):
        return len(self.docs)

    def __iter__(self):
        return iter(self.docs)

    def __repr__(self):
        return f"QueryResult({self.query.query_id}, len={len(self.docs)})"
