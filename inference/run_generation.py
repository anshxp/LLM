import argparse

import torch

from config.model_config import ModelConfig
from data.tokenizer import Tokenizer
from inference.generate import generate
from model.llm import LLM


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        raise ValueError("max-new-tokens must be non-negative")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    config = ModelConfig(vocab_size=len(tokenizer))
    model = LLM(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    prompt_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
