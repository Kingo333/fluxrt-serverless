import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
from data_tools.fluxrt_prompt import fluxrt_prompt  # noqa: E402

CASES = json.loads((Path(__file__).parent / "fluxrt_prompt_cases.json").read_text())


def test_matches_app_typescript_output():
    # Fixtures were produced by running the app's buildTryOnPrompt.ts.
    for inp, expected in CASES:
        assert fluxrt_prompt(inp.get("category"), inp.get("name"), inp.get("description"), inp.get("promptHint")) == expected


def test_dataset_categories_map():
    assert fluxrt_prompt("upper") == fluxrt_prompt("tops")
    assert fluxrt_prompt("dress") == fluxrt_prompt("dresses")
