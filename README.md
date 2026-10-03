# systemone-model-api

Authenticated local REST API for Laya and Cloudflare's [Clef and Clef-flash](https://huggingface.co/Cloudflare/clef). One model is loaded per worker at startup. Requests stay local; this is not a proxy for Workers AI.

## Run

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --frozen
export SYSTEMONE_API_KEY='<your-api-key>'
export SYSTEMONE_MODEL=clef-flash
uv run --frozen systemone-model-api
```

The service listens on port 8000. For a different bind address or port:

```sh
uv run --frozen python -m uvicorn systemone_model_api.api:app --host 127.0.0.1 --port 8001
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `SYSTEMONE_API_KEY` | Required | Bearer token for `/predict`; startup fails if missing or empty. |
| `SYSTEMONE_MODEL` | `laya` | Resident model: `laya`, `clef`, or `clef-flash`. Unknown values fail startup. |
| `SYSTEMONE_DEVICE` | Automatic | PyTorch device, such as `cuda`, `cuda:1`, or `cpu`. Clef defaults to CUDA when available, otherwise CPU; Laya uses its own device selection. |
| `HF_HOME` | Hugging Face default | Cache directory for downloaded model files. |

Clef downloads and executes Cloudflare's official custom inference code, pinned to the same release revision as its weights in `backends.py`. Clef uses BF16 weights and a 65,536-token context limit. The backbone weights alone require approximately **55 GB for Clef** or **19 GB for Clef-flash**, plus memory for the decision head and inference. Neither fits a 16 GB GPU in this configuration. There is no automatic quantization or CPU offloading; select a device with enough memory. Each additional worker loads another copy.

## API

- `GET /health`: unauthenticated liveness response, `{"status":"ok"}`. Startup completes only after the model loads.
- `POST /predict`: requires `Authorization: Bearer <SYSTEMONE_API_KEY>`.
- `GET /docs`: interactive API documentation.
- `GET /openapi.json`: generated schema, also checked in as `openapi.json`.

```sh
curl http://localhost:8000/predict \
  -H "Authorization: Bearer $SYSTEMONE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "clef-flash",
    "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {
      "urgent": {
        "type": "noul",
        "instructions": "Is this support request urgent?"
      },
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
          "billing": "Payments and invoices",
          "technical": "Outages and errors",
          "sales": "Plans and upgrades"
        }
      },
      "severity": {
        "type": "score",
        "instructions": "How severe is the customer impact?",
        "criteria": ["No impact", "Minor", "Major", "Critical"]
      }
    }
  }'
```

`model` is optional. When supplied, it must match `SYSTEMONE_MODEL`; it does not load or switch models per request. A mismatch returns 422.

`state` accepts text, a JSON object, or a JSON array, limited to 50,000 UTF-8 bytes after serialization. `questions` accepts up to 100 named questions. Each requires non-empty `instructions`:

- `noul`: probability of true. Optional `criteria` is a mapping with `true` and/or `false` descriptions, not a list.
- `choice`: named options with optional descriptions, or a list of unique option names. The list shorthand works with all backends.
- `score`: ordered level descriptions, lowest first; the answer is a probability-weighted, zero-indexed score.

Responses preserve the model's native `model`, `answers`, and `usage` fields. Choice answers include per-option probabilities and confidence; score answers also include a legend. Clef's boolean answer contains `type` and `noul`; Laya also supplies its own confidence and action fields and reports `model: "laya-rl-agent"`.

### Images

Clef and Clef-flash accept an optional `images` array. Each entry is either a base64 data URL (`data:image/png;base64,...`) or an object with `content_type` and `base64` fields. Supported types are PNG, JPEG, and WebP. Limits: four images, 4 MiB of decoded file bytes and 16 megapixels per image, 8 MiB of decoded file bytes total. Remote URLs and filesystem paths are not accepted. Invalid images and images sent to Laya return 422. Video inputs are not exposed by this REST endpoint.

## Docker

```sh
docker build -t systemone-model-api .
docker run --rm --gpus all -p 8000:8000 \
  -e SYSTEMONE_API_KEY -e SYSTEMONE_MODEL=clef-flash \
  -v systemone-model-cache:/data/hf \
  systemone-model-api
```

GPU containers require the NVIDIA Container Toolkit and enough GPU memory for the selected model. The Hugging Face cache is mounted separately so weights survive restarts.

## Development

```sh
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python scripts/generate_openapi.py > openapi.json
```

Schema generation does not require an API key, download weights, or load a model.

The distribution and executable are `systemone-model-api`; the Python package is `systemone_model_api`. Existing deployments must use the renamed package, command, import path, and `SYSTEMONE_API_KEY` environment variable. There are no legacy naming aliases. The `/predict` and `/health` paths remain unchanged.
