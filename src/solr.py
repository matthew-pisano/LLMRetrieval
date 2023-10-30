import time

import requests
from subprocess import Popen, PIPE

from src.query import QueryResult, Doc, Query


class Solr:

    def __init__(self, collection_name: str, port: int):
        self.collection_name = collection_name
        self.port = port
        self.base_url = f"http://localhost:{port}/solr/{collection_name}"

    def start(self):
        try:
            requests.get(f"http://localhost:{self.port}")
        except requests.exceptions.ConnectionError:
            self._execute(['start', '-p', str(self.port)])
            time.sleep(3)

    def stop(self):
        try:
            requests.get(f"http://localhost:{self.port}")
            self._execute(['stop'])
        except requests.exceptions.ConnectionError:
            ...

    def query(self, query: Query, rows=10):
        query_url = f"{self.base_url}/select"
        response = requests.get(query_url, params={"fl": "docno,score,doctext", "q": f"doctext:({query.query_text})", "rows": rows, "sort": "score desc"})
        if response.status_code == 200:
            docs = [Doc(doc["docno"], doc["doctext"], doc["score"]) for doc in response.json()["response"]["docs"]]
            return QueryResult(query, docs)
        raise RuntimeError(f"Received status code {response.status_code} from solr with response: {response.content}")

    @staticmethod
    def load_batch_queries(batch_query_file: str):

        queries = []
        with open(batch_query_file, "r") as file:
            line = file.readline()
            while line:
                if len(line) > 1:
                    queries.append(Query(*line.strip("\n").split(":::")))
                line = file.readline()

        return queries

    @staticmethod
    def _execute(args: list[str]):

        process = Popen(['solr'] + args, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        stdout, stderr = stdout.decode(), stderr.decode()
        if process.returncode != 0:
            raise RuntimeError(f"solr exited with code {process.returncode} and error: {stderr}")

        return stdout, stderr
