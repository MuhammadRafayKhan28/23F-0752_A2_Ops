import io

from conftest import png_bytes


def upload(client, data, name="img.png"):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["database"] is True


def test_predict_logs_to_history(client):
    r = upload(client, png_bytes())
    assert r.status_code == 200 and b"AI-generated" in r.data
    hist = client.get("/api/history").get_json()
    assert len(hist) == 1 and hist[0]["label"] == "AI-generated" and hist[0]["model_name"] == "fake"
    assert b"img.png" in client.get("/history").data


def test_wrong_type_friendly_error(client):
    r = upload(client, b"just text", "a.txt")
    assert r.status_code == 415 and b"Upload rejected" in r.data


def test_undecodable_friendly_error(client):
    r = upload(client, b"GIF89a" + b"x" * 100, "fake.gif")
    assert r.status_code == 422


def test_too_large_friendly_error(client):
    r = upload(client, png_bytes() + b"0" * (3 * 1024 * 1024), "huge.png")
    assert r.status_code == 413 and b"too large" in r.data.lower()


def test_failed_uploads_not_logged(client):
    upload(client, b"x", "a.txt")
    assert client.get("/api/history").get_json() == []
