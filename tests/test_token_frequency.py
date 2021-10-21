import unittest

from parameterized import parameterized

from data.load_20newsgroups import load_20newsgroups
from text_clf.token_frequency import get_token_frequency


class TestTokenFrequency(unittest.TestCase):
    """Class for testing token frequency."""

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
    def test_token_frequency(self, path_to_config) -> None:
        """Testing get_token_frequency function."""

        print(f"Config: {path_to_config}\n")

        token_frequency = get_token_frequency(path_to_config=path_to_config)

        self.assertEqual(len(token_frequency), 130107)
        self.assertEqual(
            token_frequency.most_common(3),
            [("the", 146532), ("to", 75064), ("of", 69034)],
        )

    @parameterized.expand(
        [
            (
                "tests/config/config_russian.yaml",
                68423,
                [("не", 10301), ("на", 7003), ("что", 5986)],
            ),
            (
                "tests/config/config_russian_pymorphy2.yaml",
                35104,
                [("не", 10301), ("на", 7003), ("что", 6417)],
            ),
        ]
    )
    def test_token_frequency_russian(
        self, path_to_config, token_frequency_len, token_frequency_most_common_3
    ) -> None:
        """Testing get_token_frequency function."""

        print(f"Config: {path_to_config}\n")

        token_frequency = get_token_frequency(path_to_config=path_to_config)

        self.assertEqual(len(token_frequency), token_frequency_len)
        self.assertEqual(
            token_frequency.most_common(3),
            token_frequency_most_common_3,
        )


if __name__ == "__main__":
    unittest.main()
