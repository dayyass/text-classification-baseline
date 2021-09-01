import traceback

from .config import get_config
from .train import _train
from .utils import close_logger, get_argparse, get_logger


def train(path_to_config: str) -> None:
    """
    Function to train baseline model with exception handler.

    :param str path_to_config: path to config.
    """

    # load config
    config = get_config(path_to_config=path_to_config)

    # get logger
    logger = get_logger(path_to_logfile=config["path_to_save_logfile"])

    try:
        _train(
            config=config,
            logger=logger,
        )
    except:  # noqa
        close_logger(logger)

        print(traceback.format_exc())


def main() -> int:
    """
    Main function to train baseline model.

    :return: exit code.
    :rtype: int
    """

    # argument parser
    parser = get_argparse()
    args = parser.parse_args()

    # train
    train(path_to_config=args.path_to_config)

    return 0


if __name__ == "__main__":
    exit(main())
