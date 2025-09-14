# code_review_app

This project now includes a small `config.py` and `logger.py` module.

Configuration:

- `config.py` exposes a `settings` object which reads a few environment variables:

  - `LLM_PROVIDER` (default: `openai`)
  - `LLM_API_KEY`
  - `REQUEST_TIMEOUT` (seconds, default: `30`)
  - `CACHE_ENABLED` (default: `1`)

- `logger.py` exports `get_logger(name)` which returns a configured stdout logger.

Set environment variables before running the app (example):

```bash
export LLM_API_KEY="sk-..."
export REQUEST_TIMEOUT=60
```
