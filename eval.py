import json
import argparse
import math
from statistics import mean
from chatbot import (dense_retriever, bm25_retriever, hybrid_retriever)

EVAL_FILE = "eval_dataset.json"
TOP_K = 4

def load_dataset():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_chunk_id(doc):
    return doc.metadata.get("chunk_id")

def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_ids = retrieved_ids[:k]
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    correct = sum(1 for chunk_id in retrieved_ids if chunk_id in relevant_set)
    return correct / len(retrieved_ids)

def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    retrieved_ids = set(retrieved_ids[:k])
    relevant_ids = set(relevant_ids)
    correct = len(retrieved_ids.intersection(relevant_ids))
    return correct / len(relevant_ids)

def hit_at_k(retrieved_ids, relevant_ids, k):
    retrieved_ids = set(retrieved_ids[:k])
    relevant_ids = set(relevant_ids)
    return float(bool(retrieved_ids.intersection(relevant_ids)))

def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_ids = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    relevant_ids = set(relevant_ids)
    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        relevance = (1 if chunk_id in relevant_ids else 0)
        if relevance:
            dcg += (relevance / math.log2(rank + 1))

    ideal_relevant = min(len(relevant_ids), k)

    if ideal_relevant == 0:
        return 0.0

    idcg = sum(1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1))

    return dcg / idcg

def evaluate_retriever(retriever, question, relevant_ids):
    documents = retriever.invoke(question)
    retrieved_ids = [get_chunk_id(doc) for doc in documents]

    return {
        "precision@4":
            precision_at_k(
                retrieved_ids,
                relevant_ids,
                TOP_K
            ),

        "recall@4":
            recall_at_k(
                retrieved_ids,
                relevant_ids,
                TOP_K
            ),

        "hit@4":
            hit_at_k(
                retrieved_ids,
                relevant_ids,
                TOP_K
            ),

        "mrr":
            reciprocal_rank(
                retrieved_ids,
                relevant_ids
            ),

        "ndcg@4":
            ndcg_at_k(
                retrieved_ids,
                relevant_ids,
                TOP_K
            ),

        "retrieved_ids":
            retrieved_ids[:TOP_K]
    }

def inspect_retrieval():
    dataset = load_dataset()
    retrievers = {
        "FAISS":
            dense_retriever,
        "BM25":
            bm25_retriever,
        "HYBRID":
            hybrid_retriever
    }

    for test_case in dataset:
        question = test_case["question"]
        print()
        print("=" * 100)
        print(f"QUESTION: {question}")
        print("=" * 100)

        candidates = {}

        for name, retriever in retrievers.items():
            documents = retriever.invoke(question)

            for doc in documents[:6]:
                chunk_id = get_chunk_id(doc)
                if chunk_id not in candidates:
                    candidates[chunk_id] = doc

        for chunk_id, doc in candidates.items():
            metadata = doc.metadata
            text = " ".join(doc.page_content.split())

            print()
            print(f"CHUNK ID: {chunk_id}")
            print(
                f"Book: "
                f"{metadata.get('book_name')}"
            )
            print(
                f"Chapter: "
                f"{metadata.get('chapter_title')}"
            )
            print(
                f"Page: "
                f"{metadata.get('page')}"
            )
            print(
                f"Text: {text[:8000]}"
            )
            print("-" * 100)

def run_evaluation():
    dataset = load_dataset()
    retrievers = {
        "FAISS":
            dense_retriever,
        "BM25":
            bm25_retriever,
        "HYBRID":
            hybrid_retriever
    }

    all_results = {
        name: []
        for name in retrievers
    }

    for test_case in dataset:
        question = test_case["question"]
        relevant_ids = test_case["relevant_chunk_ids"]

        if not relevant_ids:
            print(
                f"Skipping '{question}' "
                f"because relevant_chunk_ids "
                f"is empty."
            )
            continue

        print()
        print("=" * 100)
        print(
            f"QUESTION: {question}"
        )
        print("=" * 100)


        for name, retriever in retrievers.items():
            result = evaluate_retriever(retriever, question, relevant_ids)

            all_results[name].append(result)

            print()
            print(name)

            print(
                "Retrieved IDs:",
                result["retrieved_ids"]
            )

            print(
                "Precision@4:",
                round(
                    result[
                        "precision@4"
                    ],
                    3
                )
            )

            print(
                "Recall@4:",
                round(
                    result[
                        "recall@4"
                    ],
                    3
                )
            )

            print(
                "Hit@4:",
                round(
                    result[
                        "hit@4"
                    ],
                    3
                )
            )

            print(
                "MRR:",
                round(
                    result[
                        "mrr"
                    ],
                    3
                )
            )

            print(
                "NDCG@4:",
                round(
                    result[
                        "ndcg@4"
                    ],
                    3
                )
            )

    print()
    print("=" * 100)
    print("AVERAGE RESULTS")
    print("=" * 100)

    for name, results in all_results.items():
        if not results:
            continue

        print()
        print(name)
        print("-" * 50)

        for metric in ["precision@4", "recall@4", "hit@4", "mrr", "ndcg@4"]:
            score = mean(result[metric] for result in results)
            print(
                f"{metric}: "
                f"{score:.3f}"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect candidate chunks for manual relevance labeling."
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_retrieval()
    else:
        run_evaluation()