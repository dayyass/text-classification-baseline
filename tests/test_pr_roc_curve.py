import glob
import os
import shutil
import unittest

from parameterized import parameterized
from sklearn.metrics import auc

from data.load_20newsgroups import load_20newsgroups
from text_clf import train
from text_clf.pr_roc_curve import (
    get_precision_recall_curve,
    get_roc_curve,
    plot_precision_recall_curve,
    plot_precision_recall_f1_curves_for_thresholds,
    plot_roc_curve,
)


class TestMetricCurves(unittest.TestCase):
    """Class for testing metric curves."""

    @classmethod
    def setUpClass(cls) -> None:
        """SetUp tests with data and models."""

        load_20newsgroups()

        path_to_models_folder = "tests/models"
        if os.path.exists(path_to_models_folder):
            shutil.rmtree(path_to_models_folder)

        train(path_to_config="tests/config/config.yaml")
        train(path_to_config="tests/config/config_russian.yaml")

    @parameterized.expand(
        [
            ("tests/models/model_*",),
            ("tests/models/russian_*",),
        ]
    )
    def test_get_plot_curve(self, path_to_model_folder_pattern) -> None:
        """Testing get/plot_*_curve functions."""

        path_to_model_folder_list = glob.glob(path_to_model_folder_pattern)
        assert len(path_to_model_folder_list) == 1

        path_to_model_folder = path_to_model_folder_list[0]

        if path_to_model_folder_pattern == "tests/models/model_*":
            try:
                precision, recall, thresholds_pr = get_precision_recall_curve(
                    path_to_model_folder
                )
                fpr, tpr, thresholds_roc = get_roc_curve(path_to_model_folder)
            except AssertionError:
                self.assertTrue(True)
            except:  # noqa
                self.assertTrue(False)
        else:
            precision, recall, thresholds_pr = get_precision_recall_curve(
                path_to_model_folder
            )
            fpr, tpr, thresholds_roc = get_roc_curve(path_to_model_folder)

            self.assertNotEqual(len(thresholds_pr), len(thresholds_roc))

            pr_auc = auc(recall, precision)
            roc_auc = auc(fpr, tpr)

            self.assertGreater(pr_auc, 0.95)
            self.assertGreater(roc_auc, 0.98)

            # plot
            display_pr = plot_precision_recall_curve(precision, recall)
            display_roc = plot_roc_curve(fpr, tpr)
            display_prf = plot_precision_recall_f1_curves_for_thresholds(
                precision, recall, thresholds_pr
            )

            display_pr.plot()
            display_roc.plot()
            display_prf.plot()

    @classmethod
    def tearDownClass(cls) -> None:
        """tearDown tests with models remove."""

        path_to_models_folder = "tests/models"
        if os.path.exists(path_to_models_folder):
            shutil.rmtree(path_to_models_folder)


if __name__ == "__main__":
    unittest.main()
