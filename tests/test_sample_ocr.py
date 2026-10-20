from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "angled_page.png"


def test_photographed_sample_is_flattened_and_read():
    client = TestClient(app)
    response = client.post(
        "/api/read",
        files={"file": ("angled_page.png", SAMPLE.read_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["page_found"] is True
    assert "PAIGE" in body["text"]
    assert "vision model" in body["text"]
    assert len(body["lines"]) >= 4
    assert body["preview"].startswith("data:image/jpeg;base64,")
