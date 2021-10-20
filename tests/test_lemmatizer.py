import unittest

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

    def test_pymorphy2(self) -> None:
        """Testing pymorphy2 lemmatizer."""

        token = "стали"
        lemma = self.lemmatizer(token)  # type: ignore
        self.assertEqual(lemma, "стать")

        token = "думающему"
        lemma = self.lemmatizer(token)  # type: ignore
        self.assertEqual(lemma, "думать")

    def test_preprocessor(self) -> None:
        """Testing pymorphy2 lemmatizer."""

        sentence = "стали"
        preprocessed_sentence = self.preprocessor(sentence)  # type: ignore
        self.assertEqual(preprocessed_sentence, "стать")

        sentence = "думающему"
        preprocessed_sentence = self.preprocessor(sentence)  # type: ignore
        self.assertEqual(preprocessed_sentence, "думать")

        sentence = "стали думающему"
        preprocessed_sentence = self.preprocessor(sentence)  # type: ignore
        self.assertEqual(preprocessed_sentence, "стать думать")
