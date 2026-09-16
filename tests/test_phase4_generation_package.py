def test_generation_function_is_importable():
    from inference.generate import generate

    assert generate.__name__ == "generate"
