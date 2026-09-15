def test_inference_package_exports_generation_module():
    from inference import generate

    assert callable(generate.generate)
