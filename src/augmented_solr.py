from tqdm import tqdm

from src.models import Model
from src.query import Query, QueryResult
from src.solr import Solr
from src.utils import strip_stopwords


class AugmentedSolr(Solr):
    """An augmented version of Solr with enhanced abilities from an LLM"""

    def __init__(self, collection_name: str, port: int, model: Model):
        """
        Args:
            collection_name: The name of the collection that has been indexed by Solr
            port: The port that this system is running on
            model: The model to use for the augmented retrieval abilities"""

        super().__init__(collection_name, port)
        self.model = model

    def query(self, query: Query, expand_query=True, term_reweighting=True, filter_relevant=True, rows=10):
        """Submits a query to the model and returns the resulting documents

        Args:
            query: The query string to use
            expand_query: Whether to expand the query before retrieving
            term_reweighting: Whether to re-weight query terms before retrieving
            filter_relevant: Whether to filter out non-relevant documents using relevance feedback
            rows: The number of documents to retrieve
        Returns:
            A list of documents that are intended to be relevant to the query"""

        results = super().query(query, rows=rows if not filter_relevant else 2*rows)

        query = self.reformulate_query(query, results, expand_query=expand_query, term_reweighting=term_reweighting)
        results = super().query(query, rows=rows if not filter_relevant else 2*rows)

        if filter_relevant:
            results = self.relevance_feedback(query, results, target_results=rows)

        return results

    def reformulate_query(self, query: Query, original_result: QueryResult, expand_query: bool, term_reweighting: bool, input_doc_tokens=500, pseudo_rel_docs=2, pseudo_non_rel_docs=3):
        """Uses several methods to reformulate a user's query including term re-weighting and query expansion

        Args:
            query: The user query to reformulate
            original_result: The original result from the query
            expand_query: Whether to perform a query expansion
            term_reweighting: Whether to perform term re-weighting
            input_doc_tokens: The number of document tokens to send to the model at a time
            pseudo_rel_docs: The top N retrieved documents which should be considered pseudo-relevant
            pseudo_non_rel_docs: The bottom N retrieved documents which should be considered pseudo-non-relevant
        Returns:
            A Query object containing the reformulated query"""

        if not expand_query and not term_reweighting:
            return query

        rel_keywords = set()
        non_rel_words = set()
        query_text = query.query_text

        if term_reweighting:
            # Gather impactful keywords from non-relevant documents
            for result in original_result[-pseudo_non_rel_docs:]:
                non_rel_words.update(strip_stopwords(result.doctext).split(" "))

        # Gather impactful keywords from relevant documents
        for doc in tqdm(original_result[0:pseudo_rel_docs], desc="Processing query result"):
            model_result = self.model.keywords_by_document(strip_stopwords(doc.doctext), input_doc_tokens=input_doc_tokens)
            stripped_keywords = [strip_stopwords(kw) for kw in model_result]
            rel_keywords.update(kw for kw in stripped_keywords if kw not in non_rel_words)

        if term_reweighting:
            query = self.reweight_query_terms(query, rel_keywords, non_rel_words)

        if expand_query:
            query = self.expand_query(query, rel_keywords)

        return Query(query.query_id, query_text + " ".join(rel_keywords))

    def reweight_query_terms(self, query: Query, rel_keywords: set, non_rel_words: set, boost_factor=2, depress_factor=0.5):
        """Use Solr's query syntax to weight up any relevant terms and weight down any non-relevant terms

        Args:
            query: The query to re-weight
            rel_keywords: A set of keywords from relevant documents
            non_rel_words: A set of keywords from non-relevant documents
            boost_factor: The factor to boost the weights of relevant keywords by
            depress_factor: The factor to depress the weights of non-relevant keywords by
        Returns:
            The re-weighted query"""

        def reweight(q_text: str, kw_set: set, factor: float):
            for kw in kw_set:
                if kw in q_text:
                    splice = (q_text+" ").index(kw+" ")+len(kw)
                    q_text = q_text[:splice] + f"^{factor}" + q_text[splice:]
            return q_text

        query_text = query.query_text
        query_text = reweight(query_text, rel_keywords, boost_factor)
        query_text = reweight(query_text, non_rel_words, depress_factor)

        return Query(query.query_id, query_text)

    def expand_query(self, query: Query, rel_keywords: set):
        """Adds the given set of relevant keywords to the query

        Args:
            query: The query to expand
            rel_keywords: A set of keywords from relevant documents
        Returns:
            The expanded query"""

        filtered_keywords = [kw for kw in rel_keywords if kw not in query.query_text]
        return Query(query.query_id, query.query_text + " ".join(filtered_keywords))

    def relevance_feedback(self, query: Query, query_result: QueryResult, input_doc_tokens=500, target_results=10):
        """Filters our non-relevant documents from a list of retrieved documents by using an LLM to judge relevance

        Args:
            query: The query that was used to retrieve the documents
            query_result: The list of retrieved documents
            input_doc_tokens: The number of document tokens to send to the model at a time
            target_results: The maximum number of documents to include in the filtered results
        Returns:
            The filtered query results"""

        filtered_docs = []
        i = 1
        for doc in tqdm(query_result[:target_results], desc="Evaluating document relevance"):
            is_relevant = self.model.judge_relevance(query.query_text, doc.doctext, input_doc_tokens=input_doc_tokens)
            # Add only relevant documents to the final list
            if is_relevant:
                filtered_docs.append(doc)
            i += 1

        # Append the next best retrieved documents to pad the length to the desired value
        i = 0
        while len(filtered_docs) < target_results:
            filtered_docs.append(query_result[target_results + i])
            i += 1

        return QueryResult(query_result.query_id, filtered_docs)
