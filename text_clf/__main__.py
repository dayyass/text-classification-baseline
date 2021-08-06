from .train import train
from .utils import get_argparse


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
    train(path_to_config=args.config)

    return 0


if __name__ == "__main__":
    exit(main())
