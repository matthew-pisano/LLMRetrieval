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

        results = super().query(query, rows=rows)

        # Generate new query and gather new results
        if expand_query or term_reweighting:
            query = self.reformulate_query(query, results, expand_query=expand_query, term_reweighting=term_reweighting)
            results = super().query(query, rows=rows)

        # Filter and rerank results
        if filter_relevant:
            results = self.relevance_feedback(query, results, feedback_on_top_n=10)

        return results

    def reformulate_query(self, query: Query, original_result: QueryResult, expand_query: bool, term_reweighting: bool, input_doc_tokens=500, pseudo_rel_docs=2, pseudo_non_rel_docs=3, quiet=True):
        """Uses several methods to reformulate a user's query including term re-weighting and query expansion

        Args:
            query: The user query to reformulate
            original_result: The original result from the query
            expand_query: Whether to perform a query expansion
            term_reweighting: Whether to perform term re-weighting
            input_doc_tokens: The number of document tokens to send to the model at a time
            pseudo_rel_docs: The top N retrieved documents which should be considered pseudo-relevant
            pseudo_non_rel_docs: The bottom N retrieved documents which should be considered pseudo-non-relevant
            quiet: Whether to print out a progress bar
        Returns:
            A Query object containing the reformulated query"""

        if not expand_query and not term_reweighting:
            raise ValueError("You must choose at least one of query expansion or term reweighting.  Neither are selected")

        rel_keywords = set()
        non_rel_words = set()

        if term_reweighting:
            # Gather impactful keywords from non-relevant documents
            for result in original_result[-pseudo_non_rel_docs:]:
                non_rel_words.update(strip_stopwords(result.doctext).split(" "))

        # Gather impactful keywords from only first relevant documents
        for doc in tqdm(original_result[0:pseudo_rel_docs], desc="Processing query result", disable=quiet):
            model_result = self.model.keywords_by_document(strip_stopwords(doc.doctext), input_doc_tokens=input_doc_tokens)
            stripped_keywords = [strip_stopwords(kw) for kw in model_result]
            rel_keywords.update(kw for kw in stripped_keywords if kw not in non_rel_words)

        # Increase the weights of relevant keywords and depress the weights of non-relevant keywords in the original query
        if term_reweighting:
            query = self.reweight_query_terms(query, rel_keywords, non_rel_words)
        # Expand original query with relevant keywords extracted from the documents
        if expand_query:
            query = self.expand_query(query, rel_keywords)

        return Query(query.query_id, query.query_text)

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
            """Reweights a query on the terms in the given set by the given factor

            Args:
                q_text: The query to reweight
                kw_set: The set of keywords to reweight
                factor: The factor to reweight the selected keywords by
            Returns:
                The reweighted query"""

            new_query = q_text
            for kw in kw_set:
                if kw+" " in q_text and len(kw) > 0 and kw+" " in new_query+" ":
                    splice = (new_query+" ").index(kw+" ")+len(kw)
                    new_query = new_query[:splice] + f"^{factor}" + new_query[splice:]
            return new_query

        query_text = query.query_text
        # Perform reweighting twice on relevant and non-relevant keywords.  Helper function used to reuse code
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

    def relevance_feedback(self, query: Query, query_result: QueryResult, input_doc_tokens=500, feedback_on_top_n=10, quiet=True):
        """Filters our non-relevant documents from a list of retrieved documents and puts them at the bottom by using an LLM to judge relevance

        Args:
            query: The query that was used to retrieve the documents
            query_result: The list of retrieved documents
            input_doc_tokens: The number of document tokens to send to the model at a time
            feedback_on_top_n: The maximum number of documents to provide feedback on
            quiet: Whether to print out a progress bar
        Returns:
            The filtered query results"""

        relevant_docs = []
        non_relevant_docs = []
        i = 1

        # Judge the relevance of only the top documents.  Split these top documents into relevant and non-relevant sets
        for doc in tqdm(query_result[:feedback_on_top_n], desc="Evaluating document relevance", disable=quiet):
            is_relevant = self.model.judge_relevance(query.query_text, doc.doctext, input_doc_tokens=input_doc_tokens)
            # Update document relevance
            doc.relevant = is_relevant
            # Add only relevant documents to the final list
            if is_relevant:
                relevant_docs.append(doc)
            else:
                non_relevant_docs.append(doc)
            i += 1

        # Add the remainder of the documents to the relevant set
        for doc in query_result[feedback_on_top_n:]:
            relevant_docs.append(doc)

        # Add back in the non-relevant documents to the back of the set
        while len(non_relevant_docs) > 0:
            relevant_docs.append(non_relevant_docs.pop(0))

        return QueryResult(query_result.query_id, relevant_docs)
