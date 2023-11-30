import math


class Query:
    """A query that is given to a Solr system"""

    def __init__(self, query_id: int, query_text: str):
        """
        Args:
            query_id: The TREC id of a query
            query_text: The text of a TREC query"""

        self.query_id = query_id
        self.query_text = query_text

    def __len__(self):
        return len(self.query_text)

    def __repr__(self):
        return f"Query({self.query_id})"


class Doc:
    """A document that is retrieved from Solr"""

    def __init__(self, docno: str, doctext: str, score: float, relevant: float = -float("inf")):
        """
        Args:
            docno: The id of the document
            doctext: The text of the document
            score: The score given to the document by Solr without a reference to the ground truth
            relevant: If the document is relevant as a one or a zero. Before being set by the augmented system, it is -inf"""

        self.docno = docno
        self.doctext = doctext
        self.score = score
        self.relevant = relevant

    def __len__(self):
        return len(self.doctext)

    def __eq__(self, other):
        return self.docno == other.docno

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.docno)

    def __repr__(self):
        return f"Doc({self.docno}, {self.score})"


class QueryResult:
    """The full result from a query including statistics about its composition"""

    def __init__(self, query_id: int, result_docs: list[Doc]):
        """
        Args:
            query_id: The TREC id of the query used to retrieve this result
            result_docs: A list of retrieved documents"""

        self.query_id = query_id
        self.docs = result_docs
        self.scores = [doc.score for doc in self.docs]
        self.max_score = max(self.scores)

    @property
    def cg(self):
        """Cumulative gain using relevance judgments"""

        return sum([doc.relevant for doc in self.docs])

    @property
    def dcg(self):
        """Discounted cumulative gain using relevance judgments"""

        # This could be used in place of TREC for scoring, but TREC is more accurate due to the ground truth so this is not used
        return sum([doc.relevant/math.log2(i+2) for i, doc in enumerate(self.docs)])

    def __getitem__(self, item):
        return self.docs[item]

    def __len__(self):
        return len(self.docs)

    def __iter__(self):
        return iter(self.docs)

    def __repr__(self):
        return f"QueryResult({self.query_id}, len={len(self.docs)})"
