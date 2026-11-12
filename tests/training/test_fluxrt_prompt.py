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


def test_compact_prompt_matches_app_typescript_output():
    # Produced by running the app's buildCompactTryOnPrompt (the live FluxRT prompt), azyah-shopping #29.
    from data_tools.fluxrt_prompt import compact_prompt
    cases = json.loads((Path(__file__).parent / "fluxrt_compact_cases.json").read_text())
    for inp, expected in cases:
        assert compact_prompt(inp.get("category"), inp.get("name"), inp.get("description"), inp.get("promptHint")) == expected
    abaya = compact_prompt("modestwear", "Black Abaya")
    assert abaya.endswith("Keep the person's face, hair, hijab or head covering, body, pose and the background unchanged.")
    assert "hijab" not in compact_prompt("tops", "White Tee")
