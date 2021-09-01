# do not change the variable name!
grid_search_params = {
    "param_grid": {
        "tf-idf__ngram_range": [(1, 1), (1, 2)],
        "tf-idf__max_df": [1.0, 0.9],
        "tf-idf__min_df": [1, 15],
        "logreg__penalty": ["l1", "l2"],
        "logreg__C": [1.0, 0.5],
    },
    "scoring": "f1_weighted",
    "cv": 3,
    "error_score": 0,
    "verbose": 3,
    "n_jobs": -1,
}
