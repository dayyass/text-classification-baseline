import os
import unittest

from data.load_20newsgroups import load_20newsgroups
from text_clf.config import load_default_config
from text_clf.train import train


class TestUsage(unittest.TestCase):
    """
    Class for testing pipeline.
    """

    @classmethod
    def setUpClass(cls):
        """
        SetUp tests with config and data.
        """

        path_to_config = "config.yaml"

        if os.path.exists(path_to_config):
            os.remove(path_to_config)

        load_default_config()
        load_20newsgroups()

    def test_train(self) -> None:
        """
        Testing train function.
        """

        train()


if __name__ == "__main__":
    unittest.main()
