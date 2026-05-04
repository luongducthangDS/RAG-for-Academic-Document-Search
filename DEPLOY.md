# Cloud Deploy

## Streamlit app

- Streamlit chi dung `Qwen only` va `Qwen + Retrieval`.
- Retrieval trong app la hybrid retrieval + reranker.
- Gemini khong dung trong UI chat; Gemini chi dung cho pipeline danh gia RAGAS.

## Required environment variables for RAGAS

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- Hoac `GEMINI_API_KEYS=key1,key2,key3` de tu dong xoay key khi het quota

## Recommended environment variables

- `GEMINI_MODEL=gemini-2.5-flash`
- `LOCAL_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct`

## Run directly

```bash
streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
```

## Run with Docker

```bash
docker build -t uneti-rag .
docker run --rm -p 8501:8501 uneti-rag
```

## Smoke tests

```bash
python scripts/run_chat_cli.py --backend local
python scripts/run_ragas_evaluate.py --backend gemini --limit 5 --samples-per-batch 5
```
