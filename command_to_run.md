# Commands to Run This App

## 1. Setup virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows (cmd/PowerShell)
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure environment

```bash
cp .env.example .env
```

Then edit `.env` and set your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Get a free key at https://console.groq.com.

## 4. Add documents

Place `.pdf`, `.txt`, or `.md` files into `doc/`.

## 5. Ingest documents (run once, and again whenever `doc/` changes)

```bash
python -m src.ingestion.ingest
```

## 6. Run the app

CLI:

```bash
python -m src.main
```

Streamlit UI (alternative):

```bash
streamlit run src/ui/app.py
```

## 7. Run tests

```bash
pytest
```
