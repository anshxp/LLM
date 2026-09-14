import torch

from config.model_config import ModelConfig
from model.llm import LLM
from training.loss import language_model_loss


def test_tiny_model_can_overfit():
    torch.manual_seed(42)

    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=2, num_heads=4, dropout=0.0)
    model = LLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]] * 4)
    target_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]] * 4)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = language_model_loss(logits, target_ids)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.1
    assert final_loss < 0.2


def test_checkpoint_round_trip(tmp_path):
    from training.checkpoint import save_checkpoint, load_checkpoint

    torch.manual_seed(7)
    config = ModelConfig(vocab_size=16, context_length=8, embedding_dim=32, num_layers=1, num_heads=4, dropout=0.0)
    model = LLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "model.pt"

    input_ids = torch.randint(0, config.vocab_size, (2, config.context_length))
    model.eval()
    with torch.no_grad():
        before = model(input_ids)

    save_checkpoint(model, optimizer, 12, path)

    restored = LLM(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    step = load_checkpoint(restored, restored_optimizer, path)

    restored.eval()
    with torch.no_grad():
        after = restored(input_ids)

    assert step == 12
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-5)
