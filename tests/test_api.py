import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_reader
from app.reader import TextLine


class FakeReader:
    loaded = True

    def read(self, image_bgr):
        assert image_bgr.ndim == 3
        return [TextLine("Hello page", 0.91, (4, 4, 40, 20))]


def _png_bytes() -> bytes:
    image = Image.new("RGB", (80, 60), (250, 248, 242))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_index():
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["detector"] == "db_mobilenet_v3_large"
    page = client.get("/")
    assert page.status_code == 200
    assert "Paige" in page.text


def test_read_returns_transcript(monkeypatch):
    app.dependency_overrides[get_reader] = lambda: FakeReader()
    client = TestClient(app)
    response = client.post(
        "/api/read",
        files={"file": ("page.png", _png_bytes(), "image/png")},
    )
    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Hello page"
    assert body["lines"][0]["confidence"] == 0.91
    assert body["preview"].startswith("data:image/jpeg;base64,")
    assert body["page_found"] is False


def test_rejects_a_text_file():
    client = TestClient(app)
    response = client.post(
        "/api/read",
        files={"file": ("notes.txt", b"not a picture", "text/plain")},
    )
    assert response.status_code == 400
