import sys

from tqdm import tqdm

from src.augmented_solr import AugmentedSolr
from src.models import ManualModel, OpenAIModel, Model
from src.query import QueryResult, Query
from src.solr import Solr
from src.trec_eval import TrecEval


def aggregate_results(original_result: QueryResult, augmented_result: QueryResult, use_ground_truth=True):
    """Creates an aggregated result that is either a mix between the original and augmented results

    Args:
        original_result: The result returned from a regular Solr query
        augmented_result: The result returned from the AugmentedSolr system
        use_ground_truth: Whether to use ground truth for comparison or to just rank by scores
    Returns:
        The new, aggregated list of retrieved documents"""

    # If the results should be chosen in full or combined
    if use_ground_truth:
        original_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=[original_result])
        aggregated_docs = original_result.docs
        max_score = original_eval["map"]["all"]
        best_docs = original_result.docs

        # Attempts to replace docs within the original results with documents from the augmented results
        # A replacement is accepted if the score of the resulting documents is higher than without the new document
        for i in range(len(original_result)):
            for j in range(i, len(original_result)-1):
                aggregated_docs = [*best_docs[:j], augmented_result[i], *best_docs[j+1:]]
                aggregated_result = QueryResult(original_result.query_id, aggregated_docs)
                new_score = TrecEval.evaluate("data/groundTruth.txt", query_results=[aggregated_result])["map"]["all"]
                if new_score > max_score:
                    max_score = new_score
                    best_docs = aggregated_docs
                    break
    else:
        # Convert to set here to filter out duplicated documents
        aggregated_docs = list(set(original_result.docs + augmented_result.docs))
        aggregated_docs = sorted(aggregated_docs, key=lambda doc: doc.score, reverse=True)[:len(original_result)]

    return QueryResult(original_result.query_id, aggregated_docs)


def main():
    """The main function for running the program"""

    solr = Solr("trec", 8983)
    solr.start()
    model = OpenAIModel("gpt-3.5-turbo")
    augmented_solr = AugmentedSolr("trec", 8983, model)
    expand_query = True
    filter_relevant = True
    rows = 10

    queries = solr.load_batch_queries("data/queries.txt")

    first_query = 0
    last_query = len(queries)

    if len(sys.argv) == 2:
        raise ValueError("This program takes zero or two arguments")

    if len(sys.argv) == 3:
        if not sys.argv[1].isdigit() or int(sys.argv[1]) < 0:
            raise ValueError("The value of the first argument must be an integer >= 0")
        if not sys.argv[2].isdigit() or int(sys.argv[2]) < 0 or int(sys.argv[2]) <= int(sys.argv[1]):
            raise ValueError("The value of the second argument must be an integer >= 0 and > the first argument")

        first_query = int(sys.argv[1])
        last_query = int(sys.argv[2])

    queries = queries[first_query:last_query]

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
