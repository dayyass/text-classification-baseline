grid_search_params = {
    "param_grid": {
        "tf-idf__ngram_range": [(1, 1), (1, 2)],
        "tf-idf__max_df": [1.0, 0.95],
        "tf-idf__min_df": [1, 5],
        "logreg__penalty": ["l1", "l2"],
        "logreg__C": [1.0, 0.1, 0.01],
        "logreg__class_weight": [None, "balanced"],
    },
    "scoring": "f1_weighted",
    "cv": 3,
    "error_score": 0,
    "verbose": 1,  # TODO: test
    "n_jobs": -1,
}
