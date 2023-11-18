import json
import os

import openai
from dotenv import load_dotenv

load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")


class Model:
    """Base class for models"""

    def generate(self, prompt: str, **kwargs) -> str:
        ...

    def judge_relevance(self, querytext: str, doctext: str, input_doc_tokens: int = None, threshold=1):
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

        i = 0
        rating_str = ""
        while resp[i] not in ["/", "."] and i < len(resp):
            if resp[i].isdigit():
                rating_str += resp[i]
            i += 1

        return int(rating_str) >= threshold

    def keywords_by_document(self, doctext: str, input_doc_tokens: int = None, num_keywords=5):

        prompt = f"""You are an information retrieval expert.  Your goal is to find the {num_keywords} most relevant keywords in a document.
Give your answer as a JSON list of strings.
For example: ["keyword1", "keyword2", ...]

Document:
{doctext[:input_doc_tokens * 4] if input_doc_tokens is not None else doctext}

["""
        resp = self.generate(prompt).replace("\n", "")

        if not resp.startswith("["):
            resp = "[" + resp
        if not resp.endswith("]"):
            resp += "]"

        try:
            return set(json.loads(resp))
        except json.JSONDecodeError as e:
            print("Unable to decode:", resp)
            raise e

    def keywords_by_passage(self, doctext: str, input_doc_tokens: int = None, num_passages=3, kws_per_passage=3):

        prompt = f"""You are an information retrieval expert.  Your goal is to find {num_passages} passages (a few sentences) in documents that are the most relevant to the document as a whole.  For each passage, also select {kws_per_passage} keywords that are particularly relevant.
Give your answer as a list of JSON objects that contain keys passage and keywords.
For example: [{{"passage": "...", "keywords": [...]}}, {{...}}]

Document:
{doctext[:input_doc_tokens * 4] if input_doc_tokens is not None else doctext}

["""
        resp = self.generate(prompt).replace("\n", "")
        resp = resp.replace("}{", "}, {")

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
    """Manual input model"""

    def generate(self, prompt: str, accept_multiline=True, **kwargs):
        """Allows a manual response to be entered from the standard input"""

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
    """OpenAI API model"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs):
        """Generates a response from the target OpenAI model"""

        response = openai.ChatCompletion.create(model=self.model_name,
            messages=[{"role": "system", "content": "You are an information retrieval expert.  You gather relevant information from documents and output your responses in valid JSON format."},
                      {"role": "user", "content": prompt}])

        return response['choices'][0]['message']['content']
