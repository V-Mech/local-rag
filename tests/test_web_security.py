import re
import unittest

from fastapi.testclient import TestClient

from app.main import app

class WebSecurityTests(unittest.TestCase):
    def test_home_sets_session_and_rejects_disallowed_upload(self):
        with TestClient(app) as client:
            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertEqual(home.headers["x-content-type-options"], "nosniff")
            token = re.search(r'name="csrf_token" value="([^"]+)"', home.text).group(1)

            response = client.post(
                "/upload",
                data={"csrf_token": token},
                files={"file": ("blocked.exe", b"invalid", "application/octet-stream")},
            )
            self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
