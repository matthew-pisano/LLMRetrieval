from io import StringIO
from subprocess import Popen, PIPE

import pandas as pd

from src.query import QueryResult


class TrecEval:

    @classmethod
    def evaluate(cls, ground_truth_file: str, *, query_results: list[QueryResult] = None, solr_results_file: str = None):
        if not ((query_results is None) ^ (solr_results_file is None)):
            raise ValueError("Requires one of query_results or solr_results_file parameters")

        if query_results:
            solr_results_file = "tmp/query_results.txt"
            with open(solr_results_file, "w") as file:
                file.write(cls.format_query_results(query_results))

        stdout, stderr = cls._execute(['-q', ground_truth_file, solr_results_file])
        stdout = "Statistic\tQuery Id\tValue\n"+stdout

        eval_data = pd.read_csv(StringIO(stdout), delimiter="\t")
        eval_data["Statistic"] = eval_data["Statistic"].str.strip()
        eval_data["Value"] = pd.to_numeric(eval_data["Value"], errors='coerce')
        eval_data = eval_data.pivot(index="Query Id", columns="Statistic")["Value"]
        return eval_data

    @staticmethod
    def format_query_results(query_results: list[QueryResult]):
        formatted = ""

        for result in query_results:
            for i, doc in enumerate(result.docs):
                formatted += f"{result.query.query_id} Q0 {doc.docno.replace('[', '').replace(']', '')} {i+1} {doc.score} DFR\n"

        return formatted

    @staticmethod
    def _execute(args: list[str]):

        process = Popen(['trec_eval']+args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        stdout, stderr = stdout.decode(), stderr.decode()
        if process.returncode != 0:
            raise RuntimeError(f"trec_eval exited with code {process.returncode} and error: {stderr}")

        return stdout, stderr
