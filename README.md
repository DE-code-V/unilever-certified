# Hierarchical Text Classififcation with free and open-sourced LLMs
This repository demonstrates a hybrid NLP pipeline for classifying product reviews into a hierarchical taxonomy (Level 1 → Level 2 factors) using a combination of semantic embeddings and large language models (LLMs)
## 1. Approach:
* **Problem Understanding:** <br>
  Each review can map to multiple Level 1 (broad category) and Level 2 (granular subcategory) factors. <br>
    *Example:* <br>
    ```“Smells great and leaves skin clean.”``` and
    ```-> Fragrance-> Personal Likability, Feel/Finish → Fresh/Clean Feeling```
  At first on going through the data set I was assuming that I might need an `encoder-only` transformer but later part of the mail mentioned to use any LLM except BERT

* **Method:** <br>
1. Preprocess the training data and extract valid (Level 1, Level 2) taxonomy pairs.
2. Compute sentence embeddings using SentenceTransformer's `all-MiniLM-L6-v2` (to reduce number of tokens sent)
3. Use Cerebras Llama 3.3-70b via free inference API to perform few-shot
4. Shot the prompt with few examples to LLm
5. Evaluate on a holdout subset using macro-averaged precision, recall, and F1

## 2. Models Considered and selection rationale:
| Model                                        | Role                                                                                                                                                          | Reason                                                                                        |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 1. **SentenceTransformer (`all-MiniLM-L6-v2`)** | Semantic embedding model                                                                                                                                      | Lightweight, high-quality sentence-level embeddings for similarity search      |
| 2. **llama-3.3-70b (via Cerebras)**              | LLM classifier                                                                                                                                                | Free-tier accessible (1M tokens/day), strong contextual reasoning, and capable of multi-label classification |
                                                           
## 3. Output Accuracy
### Holdout Evaluation (10% of training data):
| Metric    | Score |
| --------- | ----- |
| Precision |  0.395 |
| Recall    | 0.367 |
| F1 Score  | 0.359 |

### Interpretation from accuracy:
* Reasonable precision and recall for an open-text multi-label problem.
* Most misses occurred in overlapping or subjective categories (e.g., Fragrance Strength vs. Personal Likability).
* Can be improved with increased few-shot examples and taxonomy-guided prompting.

## Misc and Notes
During experimentation, a hybrid version of the classifier was implemented, combining
* LLM-based reasoning (using llama-3.3-70b)
* Embedding-based similarity search (retrieving top-k semantically similar training examples using SentenceTransformer)

The motivation was to give the model more contextual grounding and improve few-shot performance by selecting examples dynamically based on semantic proximity
However, after testing on a holdout subset, the hybrid variant yielded slightly lower overall scores and higher variability

### Observations from this approach:
* The hybrid approach increased computational overhead (embedding comparisons + longer prompts) which increased the tokens count
* Some retrieved examples were semantically close but taxonomically mismatched, which confused the LLM
* With limited free-tier API access, iterative tuning of retrieval parameters (top_k, num_examples) was constrained

### *Closing Summary & Decision taken*:
Given the assignment scope and time constraints, the simpler taxonomy-guided few-shot LLM classifier was retained as the final submission

