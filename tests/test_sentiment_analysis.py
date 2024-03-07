import unittest
from unittest.mock import patch
import pandas as pd
from sentiment_analysis import SentimentAnalysisModel

class TestSentimentAnalysisModel(unittest.TestCase):

    def setUp(self):
        self.model = SentimentAnalysisModel('google/flan-t5-base')

    def test_construct_prompt(self):
        dialogue = "The weather is nice."
        expected_start = "Provide Sentiment for the following comment/conversation"
        result = self.model.construct_prompt(dialogue)
        self.assertTrue(result.startswith(expected_start))
        self.assertIn(dialogue, result)

    def test_predict(self):
        with patch.object(self.model.model, "generate", return_value="test sentiment") as mock_generate:
            result = self.model.predict(None, pd.DataFrame({"text": ["The weather is nice."]}))
            mock_generate.assert_called()
            self.assertIsInstance(result, str)

if __name__ == "__main__":
    unittest.main()
