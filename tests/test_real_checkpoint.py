"""Smoke-test a real trained checkpoint against representative prompts.

This test is intentionally lightweight and deterministic. It validates that a local
trained checkpoint can be loaded through the production inference entrypoint and that
generation completes for representative prompts. It does not claim semantic quality.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


CHECKPOINT_CANDIDATES = (
    Path("checkpoints/best_model.pt"),
    Path("checkpoints/phase7_run/best_model.pt"),
)

PROMPTS = (
    "What is hypertension?",
    "What are symptoms of diabetes?",
    "Explain anemia in simple words.",
)


@pytest.mark.real_model
def test_real_checkpoint_generates_representative_prompts(tmp_path):
    checkpoint = next((path for path in CHECKPOINT_CANDIDATES if path.exists()), None)
    if checkpoint is None:
        pytest.skip("No local trained checkpoint is available")

    results = []
    for prompt in PROMPTS:
        completed = subprocess.run(
            [
                sys.executable,
                "inference.py",
                "--checkpoint",
                str(checkpoint),
                "--prompt",
                prompt,
                "--max-new-tokens",
                "32",
                "--greedy",
                "--stop-at-eos",
                "--device",
                "cpu",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = completed.stdout.strip()
        assert output, f"No generation returned for prompt: {prompt}"
        assert "nan" not in output.lower()
        assert "inf" not in output.lower()
        results.append({"prompt": prompt, "output": output})

    report = {
        "checkpoint": str(checkpoint),
        "prompts": results,
        "notes": [
            "This is a generation smoke test, not a semantic-quality benchmark.",
            "Human review is required before treating outputs as medically reliable.",
        ],
    }
    report_path = tmp_path / "real_checkpoint_inference.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
