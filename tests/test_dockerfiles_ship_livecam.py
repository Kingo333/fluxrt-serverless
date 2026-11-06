"""server.py imports the livecam package, so every image that ships server.py must ship livecam/."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_server_image_copies_livecam():
    checked = 0
    for df in sorted(ROOT.glob("Dockerfile*")):
        text = df.read_text()
        if "server.py" not in text:
            continue
        checked += 1
        assert "COPY livecam /app/livecam" in text, f"{df.name} ships server.py without livecam/"
    assert checked >= 2  # Dockerfile.fluxrt and Dockerfile.livecam
