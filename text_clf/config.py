import os


def load_default_config(
    path_to_save_folder: str = ".",
    filename: str = "config.yaml",
) -> None:
    """
    Function to load default config.

    :param str path_to_save_folder: path to save folder (default: '.').
    :param str filename: filename (default: 'config.yaml').
    """

    path = os.path.join(path_to_save_folder, filename)

    config = [
        "seed: 42",
        "verbose: true",
        "path_to_save_folder: models",
        "",
        "# data",
        "data:",
        "  train_data_path: data/train.csv",
        "  valid_data_path: data/valid.csv",
        "  sep: ','",
        "  text_column: text",
        "  target_column: target_name_short",
        "",
        "# tf-idf",
        "tf-idf:",
        "  lowercase: true",
        "  ngram_range: (1, 1)",
        "  max_df: 1.0",
        "  min_df: 0.0",
        "",
        "# logreg",
        "logreg:",
        "  penalty: l2",
        "  C: 1.0",
        "  class_weight: balanced",
        "  solver: saga",
        "  multi_class: auto",
        "  n_jobs: -1",
    ]

    if os.path.exists(path):
        raise FileExistsError(f"file {path} already exists.")

    else:
        with open(path, mode="w") as fp:
            for line in config:
                fp.write(f"{line}\n")
