def test_phase4_ready_for_ci():
    from inference.generate import generate
    from train import main

    assert callable(generate)
    assert callable(main)
