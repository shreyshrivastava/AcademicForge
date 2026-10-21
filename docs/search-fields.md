# Search fields

Use **Search field** below the question box to tell AcademicForge which subject you mean. This control is separate from **Research Lens**, which chooses the kind of evidence to prioritize.

## Example: RAG and RAGAS

Enter `RAGAS` or `Rag and AS`, choose **Computer science & AI**, and press **Search**. The preview shows:

```text
RAGAS retrieval augmented generation evaluation
```

For `RAG for healthcare`, the interpreted query is:

```text
retrieval augmented generation for healthcare
```

The computing interpretation draws on the retrieval/evaluation meaning documented by [Ragas](https://docs.ragas.io/en/stable/). It is applied when you explicitly choose the computing field. Choosing **Music & arts** preserves the musical meaning; **All fields** keeps the existing broad search behavior. “RAG as a service” remains a retrieval-augmented-generation query and is not rewritten as the RAGAS evaluation framework.

## Presets and custom subjects

Available fields are **Computer science & AI**, **Music & arts**, **Medicine & health**, and **Energy & environment**. Choose **Other field…** to enter a subject of up to 120 characters. The app requires a value before searching with a custom field.

The field guides the external source query and filters fetched candidates using paper titles, abstracts, and available category metadata. RAG/RAGAS computing searches require evidence of the retrieval meaning: a neural raga-recognition paper is not accepted merely because it has a Computer Science category.

Custom subjects use whole-word matching: every nontrivial field word must appear somewhere in the paper's title, abstract, or category metadata. For example, `quantum physics` requires both words. This is a transparent keyword constraint rather than a universal scientific taxonomy; papers with missing metadata or different terminology can be excluded. Use a broader field if the result set is too small.

The selected field is not silently relaxed to fill results. If there are too few matching papers, the response suggests changing the query or field. External source failures retain their normal warnings and error handling.

## Results, caching, and plans

- Results show the original question and the field used to retrieve them.
- Changing the question, field, or lens marks existing results as needing another search. Existing results keep their original label until the new search completes.
- The field is part of the cache identity, so a music query cannot reuse the computing query's cached results.
- Research plans retain the original question, field, and interpreted meaning. Markdown exports include the field.
- Direct arXiv IDs remain direct ID lookups. An explicitly selected field still filters the returned paper.

## API

`POST /search` accepts an optional `domain` string. Omitting it preserves existing callers.

```json
{
  "query": "Rag and AS",
  "domain": "Computer science & AI",
  "categories": ["Evaluation Focused"]
}
```

The response adds `domain` and `search_query` alongside papers, source status, and warnings. Research-plan requests also accept `domain` so API clients can preserve the selected subject during generation.

Tests cover ambiguous spellings, music and computing isolation, custom fields, API validation, source-query expansion, cache separation, unchanged direct IDs, Streamlit controls, and research-plan context. A live arXiv source check returned the RAGAS evaluation paper first for the example above; Semantic Scholar was rate-limited during that check. This verifies an example search, not a new retrieval-accuracy benchmark.
