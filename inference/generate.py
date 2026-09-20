import torch


def _apply_top_k(logits, top_k):
    if top_k is None:
        return logits

    k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, k=k, dim=-1)
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    remove = cumulative_probabilities > top_p
    # Keep the first token whose inclusion reaches the nucleus threshold.
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)


def generate(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    do_sample=True,
    eos_token_id=None,
):
    """Autoregressively generate token IDs with bounded context memory.

    Args:
        model: Language model returning logits shaped [batch, sequence, vocab].
        input_ids: Initial token IDs shaped [batch, sequence].
        max_new_tokens: Maximum number of tokens to append.
        temperature: Sampling temperature. Must be positive when sampling.
        top_k: Optional number of highest-probability tokens to retain.
        top_p: Optional nucleus-sampling probability threshold in (0, 1].
        do_sample: If False, use deterministic greedy decoding.
        eos_token_id: Optional token ID that stops generation for a sequence.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if not isinstance(do_sample, bool):
        raise ValueError("do_sample must be a boolean")
    if do_sample and temperature <= 0:
        raise ValueError("temperature must be positive when sampling")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
    if eos_token_id is not None and eos_token_id < 0:
        raise ValueError("eos_token_id must be non-negative when provided")

    model.eval()
    context_length = model.embedding.position_embedding.num_embeddings
    generated = input_ids
    finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[:, -context_length:]
            logits = model(context)
            next_token_logits = logits[:, -1, :]

            if do_sample:
                next_token_logits = next_token_logits / temperature
                next_token_logits = _apply_top_k(next_token_logits, top_k)
                next_token_logits = _apply_top_p(next_token_logits, top_p)
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
                finished |= next_token.squeeze(-1).eq(eos_token_id)

            generated = torch.cat((generated, next_token), dim=1)

            if eos_token_id is not None and bool(finished.all()):
                break

    return generated
