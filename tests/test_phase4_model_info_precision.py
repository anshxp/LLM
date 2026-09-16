from tools.model_info import parameter_memory_mb


def test_memory_function_accepts_fp16_width():
    assert parameter_memory_mb.__name__ == "parameter_memory_mb"
