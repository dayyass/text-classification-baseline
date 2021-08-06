from .train import train
from .utils import get_argparse, get_config


def main() -> int:
    """
    Main function to train baseline model.

    :return: exit code.
    :rtype: int
    """

    # argument parser
    parser = get_argparse()
    args = parser.parse_args()

    # load config
    config = get_config(args.config)
    config["path_to_config"] = args.config

    # train
    train(config)

    return 0


if __name__ == "__main__":
    exit(main())
