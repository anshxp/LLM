import tools.model_info


def test_model_info_module_imports():
    assert callable(tools.model_info.main)
