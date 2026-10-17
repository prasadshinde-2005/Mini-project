import re
import unittest

from flask_app.app import app   # apne app file ka path yahan adjust karo


class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    # ---------- Test 1: home page ("/") ----------
    def test_home_page(self):
        response = self.client.get("/")

        # status code 200 hona chahiye
        self.assertEqual(response.status_code, 200)

        # <title> ke andar "Sentiment Analysis" hona chahiye
        html = response.data.decode("utf-8")
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(match, "Page mein <title> tag nahi mila")
        self.assertIn("Sentiment Analysis", match.group(1))

    # ---------- Test 2: predict ("/predict") ----------
    def test_predict_page(self):
        response = self.client.post("/predict", data={"text": "I love this movie"})

        # status code 200 hona chahiye
        self.assertEqual(response.status_code, 200)

        # result mein "Happy" ya "Sad" dikhna chahiye, warna fail
        html = response.data.decode("utf-8")
        self.assertTrue(
            "Happy" in html or "Sad" in html,
            "Response mein na 'Happy' mila na 'Sad'",
        )


if __name__ == "__main__":
    unittest.main()
