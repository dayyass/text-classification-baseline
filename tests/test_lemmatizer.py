import unittest

from parameterized import parameterized

from text_clf.lemmatizer import LemmatizerPymorphy2, Preprocessor


class TestLemmatizer(unittest.TestCase):
    """Class for testing lemmatizers."""

    @classmethod
    def setUpClass(cls) -> None:
        """SetUp tests with lemmatizer and preprocessor."""

        lemmatizer = LemmatizerPymorphy2()
        preprocessor = Preprocessor(lemmatizer)

        cls.lemmatizer = lemmatizer  # type: ignore
        cls.preprocessor = preprocessor  # type: ignore

    @parameterized.expand(
        [
            ("стали", "стать"),
            ("думающему", "думать"),
        ]
    )
    def test_pymorphy2(self, token, lemma_true) -> None:
        """Testing pymorphy2 lemmatizer."""

        lemma_pred = self.lemmatizer(token)  # type: ignore
        self.assertEqual(lemma_true, lemma_pred)

    @parameterized.expand(
        [
            ("стали", "стать"),
            ("думающему", "думать"),
            ("стали думающему", "стать думать"),
        ]
    )
    def test_preprocessor(self, sentence, preprocessed_sentence_true) -> None:
        """Testing pymorphy2 lemmatizer."""

        preprocessed_sentence_pred = self.preprocessor(sentence)  # type: ignore
        self.assertEqual(preprocessed_sentence_true, preprocessed_sentence_pred)
