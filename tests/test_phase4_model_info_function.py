from tools.model_info import parameter_count, parameter_memory_mb


def test_model_info_functions_are_callable():
    assert callable(parameter_count)
    assert callable(parameter_memory_mb)
