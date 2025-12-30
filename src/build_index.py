from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}


def load_documents(data_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    documents: list[str] = []
    metadata: list[dict[str, str]] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        documents.append(content)
        metadata.append({"path": str(path), "content": content})

    return documents, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local TF-IDF index for .txt/.md documents."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Folder to scan (defaults to ./data).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.source).expanduser().resolve() if args.source else root / "data"
    index_dir = root / "index"
    index_dir.mkdir(exist_ok=True)

    documents, metadata = load_documents(data_dir)
    if not documents:
        raise SystemExit(
            "No documents found. Add .txt or .md files and try again."
        )

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)

    with (index_dir / "vectorizer.pkl").open("wb") as handle:
        pickle.dump(vectorizer, handle)

    with (index_dir / "matrix.pkl").open("wb") as handle:
        pickle.dump(matrix, handle)

    with (index_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Indexed {len(documents)} documents into {index_dir}")


if __name__ == "__main__":
    main()
