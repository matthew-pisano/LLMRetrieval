import json
import os
import time

import openai
from dotenv import load_dotenv

load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")


class Model:
    """Base class for models"""

    def generate(self, prompt: str, **kwargs) -> str:
        """Generates a response to the prompt

        Args:
            prompt: The prompt to respond to
        Returns:
            The response to the prompt"""

        ...

    def judge_relevance(self, querytext: str, doctext: str, input_doc_tokens: int = None, threshold=1):
        """Judges a document as relevant or non-relevant based on its text and the query that retrieved it

        Args:
            querytext: The text of the query
            doctext: The text of the document
            input_doc_tokens: The number of document tokens to send to the model at a time
            threshold: The threshold of relevance based on the prompt's weighting scheme
        Returns:
            A boolean judgment of document relevance"""

        prompt = f"""You are an information retrieval expert.  Your goal is to judge the relevance of a document when compared to a query.
Output your judgment as a score from 0 to 3. Such that:
0 - not relevant at all
1 - possibly relevant
2 - relevant
3 - very relevant

For example:
Query: "Norwegian Banking"
Document: "Today, banks in Norway experienced a liquidity crisis leading the national bank to..."
Judgment: 3/3

Query: "Norwegian Banking"
Document: "Last week, several large banks in China raised interest rates on housing loans..."
Judgment: 2/3

Query: "Norwegian Banking"
Document: "The best places for petting goats in Kansas are..."
Judgment: 0/3

Query: "{querytext}"
Document: "{doctext[:input_doc_tokens * 4] if input_doc_tokens is not None else doctext}"
Judgment:"""

        resp = self.generate(prompt)

        # Extract the first number from a response of `score/total`
        i = 0
        rating_str = ""
        while resp[i] not in ["/", "."] and i < len(resp):
            if resp[i].isdigit():
                rating_str += resp[i]
            i += 1

        return int(rating_str) >= threshold

    def keywords_by_document(self, doctext: str, input_doc_tokens: int = None, num_keywords=5, retries=3):
        """Extracts the most relevant keywords of a document

        Args:
            doctext: The text of the document
            input_doc_tokens: The number of document tokens to send to the model at a time
            num_keywords: The number of keywords to extract from a document
            retries: The number of times to retry
        Returns:
            A set of the most relevant keywords in a document"""

        prompt = f"""You are an information retrieval expert.  Your goal is to find the {num_keywords} most relevant keywords in a document.
Please also stem your selected words.  For example, instead of "hurricanes", output "hurricane".
Give your answer as a JSON list of strings.
For example: ["keyword1", "keyword2", ...]

Document:
{doctext[:input_doc_tokens * 4] if input_doc_tokens is not None else doctext}

["""
        resp = self.generate(prompt).replace("\n", "").replace("...", "")

        # Add brackets in case the model does not generate them
        if not resp.startswith("["):
            resp = "[" + resp
        if not resp.endswith("]"):
            resp += "]"

        try:
            return set(json.loads(resp))
        except json.JSONDecodeError as e:
            if retries > 0:
                return self.keywords_by_document(doctext, input_doc_tokens=input_doc_tokens, num_keywords=num_keywords, retries=retries-1)
            print("Unable to decode:", resp)
            raise e

    def keywords_by_passage(self, doctext: str, input_doc_tokens: int = None, num_passages=3, kws_per_passage=3):
        """Extracts the most relevant keywords of a series of selected passages within a document

        Args:
            doctext: The text of the document
            input_doc_tokens: The number of document tokens to send to the model at a time
            num_passages: The number of passages to extract from the document
            kws_per_passage: The number of keywords to extract from a passage
        Returns:
            A set of the most relevant keywords in a document"""

        prompt = f"""You are an information retrieval expert.  Your goal is to find {num_passages} passages (a few sentences) in documents that are the most relevant to the document as a whole.  For each passage, also select {kws_per_passage} keywords that are particularly relevant.
Please also stem your selected words.  For example, instead of "hurricanes", output "hurricane".
Give your answer as a list of JSON objects that contain keys passage and keywords.
For example: [{{"passage": "...", "keywords": [...]}}, {{...}}]

Document:
{doctext[:input_doc_tokens * 4] if input_doc_tokens is not None else doctext}

["""
        resp = self.generate(prompt).replace("\n", "")
        resp = resp.replace("}{", "}, {")

        # Add brackets in case the model does not generate them
        if not resp.startswith("["):
            resp = "["+resp
        if not resp.endswith("]"):
            resp += "]"

        try:
            result = json.loads(resp)
            keywords = set()
            for passage in result:
                for kw in passage["keywords"]:
                    keywords.add(kw)
            return keywords
        except json.JSONDecodeError as e:
            print("Unable to decode:", resp)
            raise e


class ManualModel(Model):
    """Manual input model that gets generated text from the program's standard input"""

    def generate(self, prompt: str, accept_multiline=True, **kwargs):

        print("\n\n---\n" + prompt)

        if not accept_multiline:
            return input("> ")

        print("Multiline response.  Enter ':s' to submit your response")
        response = ""
        while True:
            resp = input("> ")
            if resp == ":s":
                return response
            elif resp.startswith(":") and len(resp) == 2:
                raise ValueError(f"Unrecognized command '{resp}'")

            response += resp + "\n"


class OpenAIModel(Model):
    """OpenAI API model that generates text through the OpenAI API"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, timeout=10, retries=4, **kwargs):
        try:
            response = openai.ChatCompletion.create(model=self.model_name, request_timeout=timeout,
                messages=[{"role": "system", "content": "You are an information retrieval expert.  You gather relevant information from documents and output your responses in valid JSON format."},
                          {"role": "user", "content": prompt}])
        except Exception as e:
            if retries > 0:
                time.sleep(10)
                return self.generate(prompt, timeout=timeout, retries=retries-1, **kwargs)
            raise e

        return response['choices'][0]['message']['content']
