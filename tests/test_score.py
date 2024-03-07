import unittest
from unittest.mock import MagicMock, patch
import score

class TestScore(unittest.TestCase):

    @patch("score.mlflow.pyfunc.load_model")
    def test_init(self, mock_load_model):
        score.init()
        mock_load_model.assert_called_with(score.model_path)

    @patch("score.json.dumps")
    @patch("score.model.predict", return_value=["Positive"])
    def test_run(self, mock_predict, mock_dumps):
        test_input = '{"text": "The weather is nice."}'
        score.run(test_input)
        mock_predict.assert_called()
        mock_dumps.assert_called()

if __name__ == "__main__":
    unittest.main()
