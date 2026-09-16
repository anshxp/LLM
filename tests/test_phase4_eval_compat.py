from evaluation.evaluate import evaluate


def test_evaluation_function_remains_available():
    assert callable(evaluate)
