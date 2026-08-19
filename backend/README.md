# HHG Backend

This is the FastAPI backend for the HHG Retrieval System.

## Development

Run the server with hot-reloading enabled for development:
```bash
uvicorn backend.main:app --reload
```

## Production

For production, do not use the `--reload` flag. It is recommended to run the app with multiple workers depending on CPU capacity.
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Make sure to configure `.env` according to `.env.example`.

## Saaras Speech-to-Text

The backend integrates Sarvam AI's Saaras STT model via `POST /api/voice` endpoint.

To enable Voice functionality, you must configure your API key in `.env`:
```
HHG_SAARAS_ENABLED=true
HHG_SAARAS_API_KEY=your_real_api_key_here
```

**Features**:
- Validates uploaded audio size limits (`HHG_MAX_AUDIO_MB`).
- Normalizes standard Sarvam languages `hi-IN, en-IN, bn-IN` back to backend formats.
- Forwards transcribed text cleanly into the exact same retrieval pipeline (and query cache) as standard text queries.
- Strict error mapping ensuring timeouts and malformed upstreams emit safe HTTP boundaries (`502/503/504`).

## SLM & Grounding Context Generation

Answer generation is completely disabled by default to preserve maximum speed and deterministic search capabilities. 

To enable generative grounding:
```
HHG_SLM_ENABLED=true
HHG_SLM_PROVIDER=openai
HHG_SLM_BASE_URL=https://api.openai.com/v1/chat/completions
HHG_SLM_API_KEY=your_key
HHG_SLM_MODEL=gpt-4o-mini
```

**Features:**
- **Deterministic Grounding Builders:** Limits inputs directly via `HHG_MAX_CONTEXT_CHARS` and `HHG_GROUNDING_TOP_K` boundaries to avoid token-window blowout and guarantee stability.
- **Provider Agnostic:** Uses raw `requests` against OpenAI-compatible `POST` streams for widespread `vLLM` and `Ollama` hosting.
- **Zero Hallucination Guardrails:** Analyzes SLM generation output buffers strictly detecting negative-knowledge variants (e.g. "I don't have enough information"), instantly rejecting ungrounded fallback logic to safely omit citations when the search index lacks scope.
- **Strict Architecture**: Reuses the core Retrieval cache without ever falsely caching a fabricated SLM result.

