from tqdm import tqdm

from src.models import Model
from src.query import Query, QueryResult
from src.solr import Solr
from src.utils import strip_stopwords


class AugmentedSolr(Solr):

    def __init__(self, collection_name: str, port: int, model: Model):

        super().__init__(collection_name, port)
        self.model = model

    def query(self, query: Query, expand_query=True, filter_relevant=True, rows=10):

        results = super().query(query, rows=rows if not filter_relevant else 2*rows)

        if expand_query:
            query = self.expand_query(query, results)
            results = super().query(query, rows=rows if not filter_relevant else 2*rows)

        if filter_relevant:
            results = self.relevance_feedback(query, results, target_results=rows)

        return results

    def expand_query(self, query: Query, query_result: QueryResult, input_doc_tokens=500, pseudo_rel_docs=2, pseudo_non_rel_docs=3):

        doc_keywords = set()
        query_text = query.query_text

        for doc in tqdm(query_result[0:pseudo_rel_docs], desc="Processing query result"):
            model_result = self.model.keywords_by_document(strip_stopwords(doc.doctext), input_doc_tokens=input_doc_tokens)
            doc_keywords.update([strip_stopwords(kw) for kw in model_result])

        non_rel_words = set()
        for result in query_result[-pseudo_non_rel_docs:]:
            non_rel_words.update(strip_stopwords(result.doctext).split(" "))

        for word in non_rel_words:
            if word in doc_keywords:
                doc_keywords.remove(word)

        return Query(query.query_id, query_text + " ".join(doc_keywords))

    def relevance_feedback(self, query: Query, query_result: QueryResult, input_doc_tokens=500, target_results=10):

        filtered_docs = []
        i = 1
        for doc in tqdm(query_result[:target_results], desc="Evaluating document relevance"):
            is_relevant = self.model.judge_relevance(query.query_text, doc.doctext, input_doc_tokens=input_doc_tokens)
            if is_relevant:
                filtered_docs.append(doc)
            print(f"Doc {doc.docno}: rank - {i}, score - {doc.score}, relevant? - {is_relevant}")
            i += 1

        i = 0
        while len(filtered_docs) < target_results:
            filtered_docs.append(query_result[target_results + i])
            i += 1

        return QueryResult(query_result.query_id, filtered_docs)
