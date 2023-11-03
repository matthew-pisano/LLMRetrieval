import random
from tqdm import tqdm

from src.models import ManualModel, OpenAIModel, Model
from src.query import QueryResult, Query
from src.solr import Solr
from src.trec_eval import TrecEval
from src.utils import strip_stopwords


def refine_query(query: Query, query_result: QueryResult, model: Model, input_doc_tokens=500, terms_to_add=2):
    biased_keywords = []
    query_text = query.query_text

    for doc in tqdm(query_result, desc="Processing document"):
        model_result = model.keywords_by_document(strip_stopwords(doc.doctext), input_doc_tokens=input_doc_tokens)
        for kw in model_result:
            stripped = strip_stopwords(kw)
            biased_keywords.extend([stripped]*int(doc.score))

    to_add = set()
    while len(to_add) < terms_to_add:
        choice = random.randint(0, len(biased_keywords)-1)
        to_add.add(biased_keywords[choice])

    return Query(query.query_id, query_text + " ".join(to_add))


def refine_batch(queries: list[Query], results: list[QueryResult], solr: Solr, model: Model, rows=10):

    refined_queries = []
    refined_results = []

    for q, res in tqdm(zip(queries, results), desc="Refining query", total=len(queries)):
        new_query = refine_query(q, res, model)
        refined_queries.append(new_query)
        refined_results.append(solr.query(new_query, rows=rows))

    return refined_queries, refined_results


def aggregate_results(original_result: QueryResult, refined_result: QueryResult, replace=True):

    # If the results should be chosen in full or combined
    if replace:
        original_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=[original_result])
        refined_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=[refined_result])
        aggregated_docs = refined_result if refined_eval["map"]["all"] > original_eval["map"]["all"] else original_result
    else:
        # Convert to set here to filter out duplicated documents
        aggregated_docs = list(set(original_result.docs + refined_result.docs))
        aggregated_docs = sorted(aggregated_docs, key=lambda doc: doc.score, reverse=True)[:len(original_result)]

    return QueryResult(original_result.query_id, aggregated_docs)


def main():
    solr = Solr("trec", 8983)
    solr.start()
    model = OpenAIModel("gpt-3.5-turbo")
    rows = 10

    queries = solr.load_batch_queries("data/queries.txt")[10:15]
    results = [solr.query(query, rows=rows) for query in queries]

    refined_queries, refined_results = refine_batch(queries, results, solr, model, rows=rows)

    agg_results = [aggregate_results(res, ref_res) for res, ref_res in zip(results, refined_results)]

    results_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=results)
    refined_results_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=refined_results)
    agg_results_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=agg_results)

    print("Average DCG:", sum([res.dcg for res in results]) / len(results))
    print("Average Refined DCG:", sum([res.dcg for res in refined_results]) / len(refined_results))
    print("Average Aggregated DCG:", sum([res.dcg for res in agg_results]) / len(agg_results))

    print("Eval (map, recip rank):", results_eval["map"]["all"], results_eval["recip_rank"]["all"])
    print("Refined Eval (map, recip rank):", refined_results_eval["map"]["all"], refined_results_eval["recip_rank"]["all"])
    print("Aggregated Eval (map, recip rank):", agg_results_eval["map"]["all"], agg_results_eval["recip_rank"]["all"])
    print()


if __name__ == "__main__":
    main()
