from config.model_config import ModelConfig
from model.llm import LLM


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_memory_mb(model, bytes_per_parameter=4):
    if bytes_per_parameter <= 0:
        raise ValueError("bytes_per_parameter must be positive")
    return parameter_count(model) * bytes_per_parameter / (1024 ** 2)


def main():
    config = ModelConfig()
    model = LLM(config)
    count = parameter_count(model)
    fp32_mb = parameter_memory_mb(model)
    adamw_mb = fp32_mb * 3

    print(f"Parameters: {count:,}")
    print(f"FP32 parameter memory: {fp32_mb:.1f} MB")
    print(f"Approx. FP32 AdamW model+optimizer memory: {adamw_mb:.1f} MB")
    print(f"Context length: {config.context_length}")
    print(f"Vocabulary size: {config.vocab_size}")


if __name__ == "__main__":
    main()
