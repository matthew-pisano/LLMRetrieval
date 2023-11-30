from io import StringIO
from subprocess import Popen, PIPE

import pandas as pd

from src.query import QueryResult


class TrecEval:
    """An instance of the Trec-Eval program that judges the results of TREC queries based on a ground truth"""

    @classmethod
    def evaluate(cls, ground_truth_file: str, *, query_results: list[QueryResult] = None, solr_results_file: str = None):
        """Evaluates a query based on either its results or a file of formatted Solr results

        Args:
            ground_truth_file: The file containing the ground truth data
            query_results: The results from a Solr query as an object
            solr_results_file: The results from a Solr query, saved to a file
        Returns:
            A dataframe of the evaluation for the query results"""

        if not ((query_results is None) ^ (solr_results_file is None)):
            raise ValueError("Requires one of query_results or solr_results_file parameters")

        # If not already in file form, write to a temporary file
        if query_results:
            solr_results_file = "tmp/query_results.txt"
            with open(solr_results_file, "w") as file:
                file.write(cls.format_query_results(query_results))

        stdout, stderr = cls._execute(['-q', ground_truth_file, solr_results_file])

        with open("tmp/trec_eval.txt", "w") as file:
            file.write(stdout)

        stdout = "Statistic\tQuery Id\tValue\n"+stdout

        # Parse the Trec-Eval document as a dataframe for easier manipulation
        eval_data = pd.read_csv(StringIO(stdout), delimiter="\t")
        eval_data["Statistic"] = eval_data["Statistic"].str.strip()
        eval_data["Value"] = pd.to_numeric(eval_data["Value"], errors='coerce')
        # Group by query id
        eval_data = eval_data.pivot(index="Query Id", columns="Statistic")["Value"]
        return eval_data

    @staticmethod
    def format_query_results(query_results: list[QueryResult]):
        """Formats the results of a query into a format that Trec-Eval can parse

        Args:
            query_results: The results from a query
        Returns:
            A string containing the formatted results"""

        formatted = ""

        # Formats into: <query id> Q0 <docno> <rank> <score> DFR
        for result in query_results:
            for i, doc in enumerate(result.docs):
                formatted += f"{result.query_id} Q0 {doc.docno.replace('[', '').replace(']', '')} {i+1} {doc.score} DFR\n"

        return formatted

    @staticmethod
    def _execute(args: list[str]):
        """Executes a command on the Trec-Eval program through its command line interface

        Args:
            args: The command line arguments to run
        Returns:
            The standard output and error of the command"""

        process = Popen(['trec_eval']+args+['-m', 'all_trec'], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        stdout, stderr = stdout.decode(), stderr.decode()

        if process.returncode != 0:
            raise RuntimeError(f"trec_eval exited with code {process.returncode} and error: {stderr}")

        return stdout, stderr
