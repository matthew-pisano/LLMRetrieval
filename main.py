from src.models import ManualModel
from src.solr import Solr
from src.trec_eval import TrecEval

if __name__ == "__main__":
    solr = Solr("trec", 8983)
    solr.start()
    queries = solr.load_batch_queries("data/queries.txt")

    results = [solr.query(query) for query in queries]
    model = ManualModel()
    model.parse_document(results[0][2].stripped())

    for query, result in zip(queries, results):
        print(query, result)

    results_eval = TrecEval.evaluate("data/groundTruth.txt", query_results=results)

    print(results_eval)
