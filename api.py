"""HTTP API for the trained LLM inference service."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.model_config import ModelConfig
from data.tokenizer import Tokenizer
from inference.generate import generate
from model.llm import LLM
from inference.run_generation import _validate_checkpoint_compatibility


DEFAULT_CHECKPOINT = "checkpoints/medquad_sft/best_model.pt"
DEFAULT_TOKENIZER = "data/processed/tokenizer.json"


@dataclass
class InferenceService:
    model: LLM
    tokenizer: Tokenizer
    device: torch.device
    checkpoint: str

    @classmethod
    def load(cls, checkpoint_path: str, tokenizer_path: str, device_name: str) -> "InferenceService":
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        device = torch.device(
            "cuda"
            if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
            else "cpu"
        )

        checkpoint = Path(checkpoint_path)
        tokenizer_file = Path(tokenizer_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_file}")

        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        config = ModelConfig(vocab_size=len(tokenizer))
        state = torch.load(str(checkpoint), map_location=device, weights_only=False)
        _validate_checkpoint_compatibility(state, config)

        model = LLM(config).to(device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        return cls(model=model, tokenizer=tokenizer, device=device, checkpoint=str(checkpoint))

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
        stop_at_eos: bool,
    ) -> str:
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        eos_token_id = self.tokenizer.token_to_id["<eos>"] if stop_at_eos else None

        output_ids = generate(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=not greedy,
            eos_token_id=eos_token_id,
        )
        return self.tokenizer.decode(output_ids[0].tolist())


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=80, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int | None = Field(default=50, ge=1, le=1000)
    top_p: float | None = Field(default=0.9, gt=0.0, le=1.0)
    greedy: bool = False
    stop_at_eos: bool = True


class GenerateResponse(BaseModel):
    prompt: str
    response: str
    model: str
    checkpoint: str


_service: InferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = InferenceService.load(
        checkpoint_path=os.getenv("LLM_CHECKPOINT", DEFAULT_CHECKPOINT),
        tokenizer_path=os.getenv("LLM_TOKENIZER", DEFAULT_TOKENIZER),
        device_name=os.getenv("LLM_DEVICE", "auto"),
    )
    yield
    _service = None


app = FastAPI(
    title="From-Scratch Medical LLM API",
    version="1.0.0",
    description="Inference API for the educational from-scratch medical language model.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "from-scratch-medical-llm",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    if _service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "status": "ok",
        "device": str(_service.device),
        "checkpoint": _service.checkpoint,
    }


@app.post("/v1/generate", response_model=GenerateResponse)
def generate_endpoint(request: GenerateRequest):
    if _service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        response = _service.generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            greedy=request.greedy,
            stop_at_eos=request.stop_at_eos,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateResponse(
        prompt=request.prompt,
        response=response,
        model="8.35M-parameter from-scratch medical LLM",
        checkpoint=_service.checkpoint,
    )
