import os
import unittest
import pandas as pd 
import mlflow
from mlflow.exceptions import MlflowException
import pickle 

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

         # import model
        with open("model/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)
    
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

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])],
        )

        prediction = self.model.predict(input_df)

        # input matches the vectorizer's feature count
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # one prediction per input row, 1-D output
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

        # binary output
        self.assertTrue(set(prediction.tolist()) <= {0, 1})

if __name__ == "__main__":
    unittest.main()