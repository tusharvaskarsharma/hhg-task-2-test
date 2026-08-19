# HHG Rollback Procedure

## Rollback to Last Stable Commit

If the current changes introduce a regression, roll back to the pre-optimization commit:

```bash
# 1. Record current HEAD
git rev-parse HEAD

# 2. Roll back to the last known-good commit
git revert HEAD --no-edit
# OR for a hard reset (loses all new commits):
git reset --hard 8332784f32ed1793fb2937a9cbd40c2d1164145d

# 3. Restart the server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --env-file backend/.env
```

## Key Commits

| SHA (first 8) | Description |
|---|---|
| `8332784` | Fix SLM model (qwen3.6-27b), 61/61 tests pass |
| Pre-this session | Known good state with RAG_ONLY P50=17.19ms, P95=24.43ms |

## Rollback Individual Changes

### Roll back extractive fast path (query.py)
```bash
git checkout 8332784 -- backend/routes/query.py
git checkout 8332784 -- backend/schemas/query.py
git checkout 8332784 -- backend/schemas/response.py
```

### Roll back config changes
```bash
git checkout 8332784 -- backend/config.py
```

### Roll back SLM model
```bash
# Edit backend/.env and set:
# HHG_SLM_MODEL=qwen/qwen3.6-27b
```

## Testing After Rollback

```bash
python -m pytest backend/tests/ -v --tb=short -q
python test_task10.py
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Expected: 61 tests pass, HTTP 200 on all queries.

## Artifact Rollback

**Artifacts are never modified by code changes.** The HNSW indexes, BM25 indexes, metadata.parquet, and ONNX model are read-only at runtime. No artifact rollback is needed.
