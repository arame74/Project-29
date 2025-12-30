# Personal Documentation Agent (Minimal Python)

This is a minimal, free-to-run setup for a personal QA agent that answers questions from your own files. It uses a local TF-IDF index (no paid APIs).

## Project Layout

```
.
├── data/                # Put your documentation files here (.txt, .md)
├── index/               # Generated index files (created by build_index.py)
├── src/
│   ├── build_index.py   # Builds the local index
│   └── ask.py           # Asks questions against your indexed docs
└── requirements.txt
```

## Quick Start

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Add your documents**
   - Place `.txt` or `.md` files inside the `data/` folder.

3. **Build the index**
   ```bash
   python src/build_index.py
   ```
   To scan another folder (including a local hard disk path), pass `--source`:
   ```bash
   python src/build_index.py --source /path/to/your/docs
   ```

4. **Ask a question**
   ```bash
   python src/ask.py "What is in my docs?"
   ```
   To generate a natural-language answer using your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-key"
   python src/ask.py "What is in my docs?" --answer
   ```
   You can override the model with `OPENAI_MODEL` or `--model`:
   ```bash
   export OPENAI_MODEL="gpt-4o-mini"
   python src/ask.py "What is in my docs?" --answer --model gpt-4o-mini
   ```

## Notes
- The index is stored in `index/` and can be rebuilt anytime by rerunning `build_index.py`.
- This is a minimal local setup. If you later want better answers, you can plug in an LLM while keeping the same `data/` and indexing flow.
