import unittest

from text_clf.lemmatizer import LemmatizerPymorphy2


class TestLemmatizer(unittest.TestCase):
    """Class for testing lemmatizers."""

    @classmethod
    def setUpClass(cls) -> None:
        """SetUp tests with lemmatizer and preprocessor."""

        cls.lemmatizer = LemmatizerPymorphy2()  # type: ignore

    def test_pymorphy2(self) -> None:
        """Testing pymorphy2 lemmatizer."""

        token = "стали"
        lemma = self.lemmatizer(token)  # type: ignore
        self.assertEqual(lemma, "стать")

        token = "думающему"
        lemma = self.lemmatizer(token)  # type: ignore
        self.assertEqual(lemma, "думать")
