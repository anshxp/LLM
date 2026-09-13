import torch


def generate(model, input_ids, max_new_tokens, temperature=1.0):
    """Autoregressively generate token IDs from an initial prompt."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    model.eval()
    context_length = model.embedding.position_embedding.num_embeddings

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids[:, -context_length:]
            logits = model(context)
            next_token_logits = logits[:, -1, :] / temperature
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

    return input_ids
