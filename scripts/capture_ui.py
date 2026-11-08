"""Generate real UI screenshots using Edge headless for Task 4 and Task 5."""
import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from werkzeug.serving import make_server

# ensure project root in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from webapp.app import create_app  # noqa: E402

EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 5055
BASE_URL = f"http://127.0.0.1:{PORT}"
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class ServerThread(threading.Thread):
    def __init__(self, app, port):
        super().__init__(daemon=True)
        self.server = make_server("127.0.0.1", port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


def take_screenshot(url: str, out_png: Path, width=1280, height=800):
    cmd = [
        EDGE_EXE,
        "--headless",
        "--disable-gpu",
        f"--window-size={width},{height}",
        f"--screenshot={out_png.resolve()}",
        url,
    ]
    subprocess.run(cmd, check=True)
    print(f"Captured {out_png.name} ({out_png.stat().st_size // 1024} KB)")


def main():
    db_file = ROOT / "reports" / "demo_ui.db"
    if db_file.exists():
        db_file.unlink()

    app = create_app(database_url=f"sqlite:///{db_file}")
    server = ServerThread(app, PORT)
    server.start()
    time.sleep(1.5)  # wait for server to start

    try:
        # 1. Capture Upload page
        take_screenshot(f"{BASE_URL}/", OUT_DIR / "ui_upload.png")

        # 2. Submit AI image
        ai_img = next((ROOT / "data" / "raw" / "AiArtData").glob("*.jpg"))
        with open(ai_img, "rb") as f:
            res_ai = requests.post(f"{BASE_URL}/predict", files={"image": (ai_img.name, f.read(), "image/jpeg")})
        html_ai = res_ai.text.replace('href="/static/', f'href="{BASE_URL}/static/')
        (OUT_DIR / "res_ai.html").write_text(html_ai, encoding="utf-8")
        take_screenshot(f"file:///{OUT_DIR / 'res_ai.html'}", OUT_DIR / "ui_result_ai.png")

        # 3. Submit Real image
        real_img = next((ROOT / "data" / "raw" / "RealArt").glob("*.jpg"))
        with open(real_img, "rb") as f:
            res_real = requests.post(f"{BASE_URL}/predict", files={"image": (real_img.name, f.read(), "image/jpeg")})
        html_real = res_real.text.replace('href="/static/', f'href="{BASE_URL}/static/')
        (OUT_DIR / "res_real.html").write_text(html_real, encoding="utf-8")
        take_screenshot(f"file:///{OUT_DIR / 'res_real.html'}", OUT_DIR / "ui_result_real.png")

        # Submit a couple more for a rich history table
        for extra in list((ROOT / "data" / "raw" / "RealArt").glob("*.jpg"))[1:4]:
            with open(extra, "rb") as f:
                requests.post(f"{BASE_URL}/predict", files={"image": (extra.name, f.read(), "image/jpeg")})
        for extra in list((ROOT / "data" / "raw" / "AiArtData").glob("*.jpg"))[1:3]:
            with open(extra, "rb") as f:
                requests.post(f"{BASE_URL}/predict", files={"image": (extra.name, f.read(), "image/jpeg")})

        # 4. Capture History page
        take_screenshot(f"{BASE_URL}/history", OUT_DIR / "ui_history.png", height=900)

        # 5. Capture Health JSON page
        take_screenshot(f"{BASE_URL}/health", OUT_DIR / "ui_health.png", width=900, height=450)

        # 6. Capture Failure Case 1: Wrong file type (.txt)
        res_wrong = requests.post(f"{BASE_URL}/predict", files={"image": ("notes.txt", b"plain text payload", "text/plain")})
        html_wrong = res_wrong.text.replace('href="/static/', f'href="{BASE_URL}/static/')
        (OUT_DIR / "err_wrong.html").write_text(html_wrong, encoding="utf-8")
        take_screenshot(f"file:///{OUT_DIR / 'err_wrong.html'}", OUT_DIR / "ui_err_wrong_type.png")

        # 7. Capture Failure Case 2: Undecodable / corrupt image
        res_corrupt = requests.post(f"{BASE_URL}/predict", files={"image": ("corrupted_avatar.png", b"not a png image binary data 1234567890", "image/png")})
        html_corrupt = res_corrupt.text.replace('href="/static/', f'href="{BASE_URL}/static/')
        (OUT_DIR / "err_corrupt.html").write_text(html_corrupt, encoding="utf-8")
        take_screenshot(f"file:///{OUT_DIR / 'err_corrupt.html'}", OUT_DIR / "ui_err_undecodable.png")

        # 8. Capture Failure Case 3: Oversize file (>5MB)
        big_data = b"0" * (6 * 1024 * 1024)
        res_oversize = requests.post(f"{BASE_URL}/predict", files={"image": ("huge_photo.jpg", big_data, "image/jpeg")})
        html_oversize = res_oversize.text.replace('href="/static/', f'href="{BASE_URL}/static/')
        (OUT_DIR / "err_oversize.html").write_text(html_oversize, encoding="utf-8")
        take_screenshot(f"file:///{OUT_DIR / 'err_oversize.html'}", OUT_DIR / "ui_err_oversize.png")

        print("All UI screenshots captured successfully!")

    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
