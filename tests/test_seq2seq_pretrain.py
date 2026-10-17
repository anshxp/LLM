import torch
from model.seq2seq.config import Seq2SeqConfig
from model.seq2seq.model import Seq2SeqLLM

def test_encoder_decoder_forward_and_gradients():
    config=Seq2SeqConfig(vocab_size=32,context_length=16,embedding_dim=32,num_heads=4,
                         num_encoder_layers=2,num_decoder_layers=2,dropout=0.0)
    model=Seq2SeqLLM(config)
    source=torch.randint(0,32,(2,8))
    decoder=torch.randint(0,32,(2,7))
    logits=model(source,decoder)
    assert logits.shape==(2,7,32)
    logits.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
