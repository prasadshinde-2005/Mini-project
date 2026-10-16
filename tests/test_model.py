import os
import unittest

import mlflow
from mlflow.exceptions import MlflowException


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = "prasadshinde-2005"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(
            "https://dagshub.com/prasadshinde-2005/Mini-project.mlflow"
        )

        cls.model_name = "spmodel"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        if cls.model_version is None:
            raise RuntimeError("No model with alias 'candidate' found")

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)

    @staticmethod
    def get_latest_model_version(model_name):
        client = mlflow.MlflowClient()
        try:
            mv = client.get_model_version_by_alias(model_name, "candidate")
            return mv.version
        except MlflowException:
            return None

    def test_model_load_property(self):
        self.assertIsNotNone(self.model)


if __name__ == "__main__":
    unittest.main()