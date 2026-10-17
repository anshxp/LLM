import torch
from config.model_config import ModelConfig
from model.llm import LLM
def test_encoder_decoder_forward_and_gradients():
    c=ModelConfig(vocab_size=32,context_length=16,embedding_dim=32,num_heads=4,num_encoder_layers=2,num_decoder_layers=2,dropout=0.)
    m=LLM(c); src=torch.randint(0,32,(2,8)); dec=torch.randint(0,32,(2,7)); logits=m(src,dec)
    assert logits.shape==(2,7,32); logits.mean().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
