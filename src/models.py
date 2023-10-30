import openai

from src.query import Doc


class Model:
    """Base class for models"""

    def generate(self, prompt: str, **kwargs) -> str:
        ...

    def parse_document(self, doctext: str):

        prompt = f"""You are an information retrieval expert.  Your goal is to find passages (a few sentences) in documents that are the most relevant to the document as a whole.  For each passage, also select some keywords that are particularly relevant.
Give your answer as a list of JSON objects that contain keys passage and keywords.
For example: [{{"passage": "...", "keywords": [...]}}, {{...}}]

Document:
{doctext}

["""
        return self.generate(prompt)


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
            messages=[{"role": "system", "content": "You are an information retrieval expert.  You gather relevant information from documents."},
                      {"role": "user", "content": prompt}])

        return response['choices'][0]['message']['content']
