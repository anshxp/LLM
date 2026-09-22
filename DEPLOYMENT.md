# Inference API

The project now exposes the trained checkpoint through FastAPI. The service loads the SFT checkpoint once at startup and keeps the model in memory for subsequent requests.

## Run locally

From the repository root, with the virtual environment active:

```powershell
pip install -r requirements.txt
$env:LLM_CHECKPOINT="checkpoints/medquad_sft/best_model.pt"
$env:LLM_TOKENIZER="data/processed/tokenizer.json"
$env:LLM_DEVICE="cpu"
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The API is then available at `http://localhost:8000` and interactive Swagger documentation is available at `http://localhost:8000/docs`.

## Health check

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Generate text

```powershell
$body = @{ prompt = "What is hypertension?"; max_new_tokens = 80; temperature = 0.8; top_k = 50; top_p = 0.9; greedy = $false; stop_at_eos = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/v1/generate -Method Post -ContentType "application/json" -Body $body
```

The frontend can call `POST /v1/generate` with the same JSON fields.

## Configuration

- `LLM_CHECKPOINT`: path to the trained `.pt` checkpoint. Defaults to `checkpoints/medquad_sft/best_model.pt`.
- `LLM_TOKENIZER`: path to the tokenizer JSON. Defaults to `data/processed/tokenizer.json`.
- `LLM_DEVICE`: `cpu`, `cuda`, or `auto`. Defaults to `auto`.

Model files are intentionally excluded from Git by `.gitignore`. For a cloud deployment, provide the checkpoint through the platform's persistent storage, an object-storage download step, or another artifact mechanism rather than committing the binary to the repository.

## API contract

`GET /health` returns service and model status.

`POST /v1/generate` accepts:

```json
{
  "prompt": "What is hypertension?",
  "max_new_tokens": 80,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.9,
  "greedy": false,
  "stop_at_eos": true
}
```

and returns:

```json
{
  "prompt": "What is hypertension?",
  "response": "...",
  "model": "8.35M-parameter from-scratch medical LLM",
  "checkpoint": "checkpoints/medquad_sft/best_model.pt"
}
```

This is an educational model and should not be presented as a clinical diagnostic or treatment system.
