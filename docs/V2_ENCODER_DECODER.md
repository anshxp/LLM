# LLM v2: Encoder-Decoder Transformer

This branch introduces an encoder-decoder Transformer as a new architecture while preserving the existing decoder-only implementation on `main`.

## Configuration

The default v2 configuration is:

- vocabulary: 10,000
- context length: 256
- embedding dimension: 256
- encoder layers: 3
- decoder layers: 3
- attention heads: 4
- feed-forward multiplier: 4
- dropout: 0.1

The model has 10,787,344 trainable parameters with the default configuration, i.e. approximately 10.8M / 11M parameters.

## Architecture

```text
source tokens
     |
shared token embedding + encoder positions
     |
3 x bidirectional encoder blocks
     |
encoder states
     |
     +-----------------------------+
                                   |
decoder tokens                    |
     |                            |
shared token embedding + decoder positions
     |
3 x decoder blocks                |
  - causal self-attention         |
  - encoder-decoder cross-attention <--- encoder states
  - feed-forward                  |
     |                            |
final norm                        |
     |                            |
LM head --------------------------+
     |
next-token logits
```

The encoder can attend to the complete source sequence. The decoder remains autoregressive: its self-attention is causal and it uses cross-attention to read the encoder representation.

## Checkpoint compatibility

The existing ~8.35M decoder-only checkpoint is **not directly compatible** with this architecture. The number of layers, attention structure, and parameter shapes differ, and the new model introduces encoder and cross-attention weights.

Therefore this branch should be treated as a v2 architecture experiment requiring a new training run. It should not replace the existing checkpoint until the v2 model has been trained and evaluated.

## DeepSeek memory optimization

DeepSeek-V2's Multi-head Latent Attention (MLA) is deliberately not included in this branch. MLA primarily compresses the key-value cache for efficient autoregressive inference; it is not a universal six-times reduction in total training RAM. It will be considered as a separate v3 attention experiment after the v2 architecture is validated.
