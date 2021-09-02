import os
import unittest

import yaml

from data.load_20newsgroups import load_20newsgroups
from text_clf.__main__ import train
from text_clf.config import load_default_config


class TestUsage(unittest.TestCase):
    """
    Class for testing pipeline.
    """

    @classmethod
    def setUpClass(cls) -> None:
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

        train(path_to_config="config.yaml")

    def test_train_grid_search(self) -> None:
        """
        Testing train function with grid_search.
        """

        with open("config.yaml", mode="r") as fp:
            config = yaml.safe_load(fp)

        config["grid-search"]["do_grid_search"] = True
        config["grid-search"][
            "grid_search_params_path"
        ] = "tests/hyperparams_for_tests.py"

        config_grid_search_path = "config_grid_search.yaml"

        with open(config_grid_search_path, mode="w") as fp:
            yaml.safe_dump(config, fp)

        train(path_to_config=config_grid_search_path)

    @classmethod
    def tearDownClass(cls) -> None:
        """
        TearDown after tests.
        """

        os.remove("config_grid_search.yaml")


if __name__ == "__main__":
    unittest.main()
