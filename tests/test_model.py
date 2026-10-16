import os
import unittest
import pickle

import pandas as pd
import mlflow
from mlflow.exceptions import MlflowException
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


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

        cls.new_model_name = "spmodel"
        cls.new_model_version = cls.get_candidate_model_version(cls.new_model_name)
        if cls.new_model_version is None:
            raise RuntimeError("No model with alias 'candidate' found")

        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # load vectorizer
        with open("model/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)

        # load holdout test data
    
        cls.holdout_data = pd.read_csv("data\feature_building_data\test_bow.csv")
    @staticmethod
    def get_candidate_model_version(model_name):
        client = mlflow.MlflowClient()
        try:
            mv = client.get_model_version_by_alias(model_name, "candidate")
            return mv.version
        except MlflowException:
            return None

    def test_model_load_property(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])],
        )

        prediction = self.new_model.predict(input_df)   # fixed: new_model

        # input matches the vectorizer's feature count
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # one prediction per input row, 1-D output
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

        # binary output
        self.assertTrue(set(prediction.tolist()) <= {0, 1})

    def test_model_performance(self):
        x_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new = self.new_model.predict(x_holdout)

        accuracy = accuracy_score(y_holdout, y_pred_new)
        precision = precision_score(y_holdout, y_pred_new, pos_label=1, zero_division=0)
        recall = recall_score(y_holdout, y_pred_new, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_holdout, y_pred_new)

        expected_accuracy = 0.70
        expected_precision = 0.70
        expected_recall = 0.70
        expected_auc = 0.70

        # fixed: compare the computed values, not the sklearn functions
        self.assertGreaterEqual(accuracy, expected_accuracy,
                                f"Accuracy should be at least {expected_accuracy}, got {accuracy:.3f}")
        self.assertGreaterEqual(precision, expected_precision,
                                f"Precision should be at least {expected_precision}, got {precision:.3f}")
        self.assertGreaterEqual(recall, expected_recall,
                                f"Recall should be at least {expected_recall}, got {recall:.3f}")
        self.assertGreaterEqual(auc, expected_auc,
                                f"AUC should be at least {expected_auc}, got {auc:.3f}")


if __name__ == "__main__":
    unittest.main()