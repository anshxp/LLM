import torch


@torch.no_grad()
def generate(model, input_ids, max_new_tokens):
    model.eval()

    for _ in range(max_new_tokens):
        context = input_ids[
            :, -model.embedding.position_embedding.num_embeddings:
        ]

        logits = model(context)

        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

    return input_ids