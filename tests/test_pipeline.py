import os
import shutil
import unittest

from parameterized import parameterized

from data.load_20newsgroups import load_20newsgroups
from text_clf import train


class TestUsage(unittest.TestCase):
    """Class for testing pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """SetUp tests with data."""

        load_20newsgroups()

    @parameterized.expand(
        [
            ("tests/config/config.yaml",),
            ("tests/config/config_pymorphy2.yaml",),
            ("tests/config/config_grid_search.yaml",),
        ]
    )
    def test_train(self, path_to_config) -> None:
        """Testing train function."""

        print(f"Config: {path_to_config}\n")

        train(path_to_config=path_to_config)
        self.assertTrue(True)

    def test_train_error(self) -> None:
        """Testing train function."""

        print("Config: tests/config/config_lemmatizer_error.yaml\n")

        try:
            train(path_to_config="tests/config/config_lemmatizer_error.yaml")
        except KeyError:
            self.assertTrue(True)
        except:  # noqa
            self.assertTrue(False)

    @classmethod
    def tearDownClass(cls) -> None:
        """tearDown tests with models remove."""

        path_to_models_folder = "tests/models"
        if os.path.exists(path_to_models_folder):
            shutil.rmtree(path_to_models_folder)


if __name__ == "__main__":
    unittest.main()
