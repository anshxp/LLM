import torch


def generate(model, input_ids, max_new_tokens, temperature=1.0, top_k=None):
    """Autoregressively generate token IDs with bounded context memory."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")

    model.eval()
    context_length = model.embedding.position_embedding.num_embeddings

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids[:, -context_length:]
            logits = model(context)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, k=k, dim=-1)
                threshold = values[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold,
                    float("-inf"),
                )

            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids
