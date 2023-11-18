from tqdm import tqdm

from src.augmented_solr import AugmentedSolr
from src.models import ManualModel, OpenAIModel, Model
from src.query import QueryResult, Query
from src.solr import Solr
from src.trec_eval import TrecEval


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
    augmented_solr = AugmentedSolr("trec", 8983, model)
    expand_query = False
    filter_relevant = True
    rows = 10

    queries = solr.load_batch_queries("data/queries.txt")[15:20]

    results = [solr.query(query, rows=rows) for query in queries]
    augmented_results = [augmented_solr.query(query, expand_query=expand_query, filter_relevant=filter_relevant, rows=rows) for query in queries]

    agg_results = [aggregate_results(res, aug_res) for res, aug_res in zip(results, augmented_results)]

    results_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=results)
    refined_results_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=augmented_results)
    agg_results_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=agg_results)

    print("Average DCG:", sum([res.dcg for res in results]) / len(results))
    print("Average Refined DCG:", sum([res.dcg for res in augmented_results]) / len(augmented_results))
    print("Average Aggregated DCG:", sum([res.dcg for res in agg_results]) / len(agg_results))

    print("Eval (map, recip rank):", results_eval["map"]["all"], results_eval["recip_rank"]["all"])
    print("Refined Eval (map, recip rank):", refined_results_eval["map"]["all"], refined_results_eval["recip_rank"]["all"])
    print("Aggregated Eval (map, recip rank):", agg_results_eval["map"]["all"], agg_results_eval["recip_rank"]["all"])
    print()


if __name__ == "__main__":
    main()
