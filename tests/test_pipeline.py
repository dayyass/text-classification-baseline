import unittest

from data.load_20newsgroups import load_20newsgroups
from text_clf.__main__ import train


class TestUsage(unittest.TestCase):
    """
    Class for testing pipeline.
    """

    @classmethod
    def setUpClass(cls):
        """
        SetUp tests with data.
        """

        load_20newsgroups()

    def test_train(self) -> None:
        """
        Testing train function.
        """

        train()


if __name__ == "__main__":
    unittest.main()
