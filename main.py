import random
from tqdm import tqdm

from src.models import ManualModel, OpenAIModel, Model
from src.query import QueryResult, Query
from src.solr import Solr
from src.trec_eval import TrecEval
from src.utils import strip_stopwords


def refine_query(query_result: QueryResult, model: Model, input_doc_tokens=500, terms_to_add=2):
    biased_keywords = []
    query_text = query_result.query.query_text

    for doc in query_result:
        model_result = model.keywords_by_document(strip_stopwords(doc.doctext), input_doc_tokens=input_doc_tokens)
        for kw in model_result:
            stripped = strip_stopwords(kw)
            biased_keywords.extend([stripped]*int(doc.score))

    to_add = set()
    while len(to_add) < terms_to_add:
        choice = random.randint(0, len(biased_keywords)-1)
        to_add.add(biased_keywords[choice])

    return Query(query_result.query.query_id, query_text + " ".join(to_add))


if __name__ == "__main__":
    solr = Solr("trec", 8983)
    solr.start()
    model = OpenAIModel("gpt-3.5-turbo")
    rows = 10

    queries = solr.load_batch_queries("data/queries.txt")[:5]
    results = [solr.query(query, rows=rows) for query in queries]

    refined_queries = []
    refined_results = []

    for res in tqdm(results, desc="Refining query"):
        new_query = refine_query(res, model)
        refined_queries.append(new_query)
        refined_results.append(solr.query(new_query, rows=rows))

    results_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=results)
    refined_results_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=refined_results)

    print("Average DCG:", sum([res.dcg for res in results]) / len(results))
    print("Average Refined DCG:", sum([res.dcg for res in refined_results]) / len(refined_results))

    print("Eval:", results_eval["map"])
    print("Refined Eval:", refined_results_eval["map"])
