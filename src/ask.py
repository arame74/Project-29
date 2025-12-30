from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from urllib import request

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_index(index_dir: Path):
    with (index_dir / "vectorizer.pkl").open("rb") as handle:
        vectorizer = pickle.load(handle)

    with (index_dir / "matrix.pkl").open("rb") as handle:
        matrix = pickle.load(handle)

    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    return vectorizer, matrix, metadata


def search(query: str, vectorizer, matrix, metadata, top_k: int = 3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    if scores.size == 0:
        return []

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append(
            {
                "score": score,
                "path": metadata[idx]["path"],
                "content": metadata[idx]["content"],
            }
        )
    return results


def build_context(results: list[dict[str, str]]) -> str:
    sections = []
    for result in results:
        sections.append(f"Source: {result['path']}\n{result['content']}")
    return "\n\n".join(sections)


def generate_answer(question: str, context: str, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            },
        ],
        "temperature": 0.2,
    }

    req = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions against your local docs.")
    parser.add_argument("query", nargs="?", help="Question to ask")
    parser.add_argument("--top", type=int, default=3, help="Number of results to show")
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Generate an answer using the OpenAI API.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model to use when --answer is provided.",
    )
    args = parser.parse_args()

    query = args.query or input("Question: ").strip()
    if not query:
        raise SystemExit("Please provide a question.")

    root = Path(__file__).resolve().parents[1]
    index_dir = root / "index"

    if not (index_dir / "vectorizer.pkl").exists():
        raise SystemExit("Index not found. Run: python src/build_index.py")

    vectorizer, matrix, metadata = load_index(index_dir)
    results = search(query, vectorizer, matrix, metadata, top_k=args.top)

    if not results:
        print("No matching documents found.")
        return

    print("Top matches:")
    for result in results:
        print(f"- {result['path']} (score: {result['score']:.3f})")

    if args.answer:
        context = build_context(results)
        answer = generate_answer(query, context, args.model)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
