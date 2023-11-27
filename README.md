# LLM Augmented Retrieval

## Overview

The aim of this project is to augment the Solr information retrieval system with the capabilities of a large language model.  This model is GPT-3.5.  The combined system performs two types of actions: query reformulation and giving relevance feedback on retrieved documents.

## Quick Start

This project requires several dependencies before it is ready to run:

* A valid Python 3 environment with the packages from the [requirements.txt](requirements.txt) file installed.
  * This should ideally be Python 3.10 or above, but older versions may still work.
* A valid OpenAI account is needed to make API calls to GPT.
* A `.env` file in the root directory of this project.  Within this file, the keys of `OPENAI_API_KEY` and `OPENAI_ORGANIZATION` should be filled in with valid credentials.
  * An example [env](env) already exists to make this process more straightforward.  Simply fill in the two keys that are already in the file and rename the file to `.env` by adding a period to the beginning.

To run the system, run the [main.py](main.py) file with:

```bash
python main.py <FIRST_QUERY> <LAST_QUERY>
```

where the arguments are both integers representing the index of the first TREC query to run the system on and the last query to run the system on.  If both arguments are not provided, all TREC queries will be used.

### Stopword Removal

Many of the functions of this augmented retrieval system require the removal of stopwords from text.  For this, the `nltk` library is used.  Each stopword from `nltk.corpus.stopwords` is then replaced with a space in the given text.

## Query Reformulation

Queries are reformulated here by using two methods: term expansion and term reweighting.  These operations can be either be performed individually or can both be performed.  When both improvements are requested, the term reweighting is performed first to make sure only the original query terms are affected.  Both of these require a list of retrieved documents from the original query.

### Extracting Relevant Keywords

An important component of both term reweighting and query expansion is a set of relevant keywords.  Traditionally, these are extracted by removing the stopwords from some of the top documents and putting them into a set.  This can be improved by using an LLM, however.  Here, the model is asked to select only a few of the most relevant keywords from a document as a whole or from a set of relevant passages.  These more relevant keywords or phrases are then added into a set after any stopwords are removed.

### Term Reweighting

This process requires a set of relevant keywords and a set of non-relevant keywords.  The non-relevant set is produced first.  It utilizes a technique from pseudo-relevance feedback and selects the last few documents in the original retrieved list.  These documents are then stripped of stopwords and added to a set.  Second, the relevant keywords are extracted using the process described above.  Relevant words that are already in the non-relevant set are not considered.  For the reweighting itself, the Solr syntax of `{term}^{weight}` is used, where relevant words are boosted by some factor and non-relevant keywords are repressed by some factor.

### Query Expansion

For query expansion, the same relevant terms from above are reused to expand the original query.  These relevant keywords are appended to the query with a space, provided that they are not already in the query.

## Relevance Feedback

During this process, documents deemed non-relevant by the LLM are removed from the list of retrieved documents.  After getting a new list of documents from the reformulated query, each is judged on a three point scale from zero (not relevant at all), to three (very relevant).  Any document that scores a zero is removed from the list.  If relevance feedback is selected, the system will automatically generate double the required number of documents from the reformulated query as overflow.  When a document is removed from the list, it is padded at the end with the next highest document from the overflow section.  Since only the first half of the documents are judged, the length of the returned list will always be the requested length.  This is because there are enough documents to fill in evan a completely replaced list.

## Prompting

The prompts given to the model utilize several prompt engineering techniques to ensure the most useful responses.  One technique used is *framing*.  This technique places the model into a particular role, encouraging it to behave as if it is well-trained at a particular task.  For example, *"You are an information retrieval expert"*.  Another technique used is few-shot priming.  This strategy gives the model one or more examples of desired responses.  This is especially useful in generating relevance judgements.