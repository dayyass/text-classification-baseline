import traceback
from typing import Dict, Tuple

from sklearn.pipeline import Pipeline

from .config import get_config
from .train import _train
from .utils import close_logger, get_argparse, get_logger


def train(path_to_config: str) -> Tuple[Pipeline, Dict[int, str]]:
    """Function to train baseline model with exception handler.

    Args:
        path_to_config (str): Path to config.

    Returns:
        Tuple[Pipeline, Dict[int, str]]:
        Model pipeline (tf-idf + logreg) and target names mapping. Both None if any exception occurred.
    """

    # load config
    config = get_config(path_to_config=path_to_config)

    # get logger
    logger = get_logger(path_to_logfile=config["path_to_save_logfile"])

    try:
        pipe, target_names_mapping = _train(
            config=config,
            logger=logger,
        )

    except:  # noqa
        close_logger(logger)

        print(traceback.format_exc())

        pipe, target_names_mapping = None, None  # type: ignore

    return pipe, target_names_mapping


def main() -> int:
    """Main function to train baseline model.

    Returns:
        int: Exit code.
    """

    # argument parser
    parser = get_argparse()
    args = parser.parse_args()

    # train
    _ = train(path_to_config=args.path_to_config)

    return 0


if __name__ == "__main__":
    exit(main())
