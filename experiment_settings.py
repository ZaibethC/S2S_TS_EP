"""Define settings for experimental design and organization.
"""

import numpy as np

__date__ = "12 December 2022"

experiments = {

    "exp000": {
        # experiment parameters
        "model_type": "logistic_regression",  # logistic_regression, random_forest
        "predictand": "label_cyclogensis_boolean",
        "years": (1979, 2021),
        "storm_type": "first_tropicalstorm",  # first_tropicalstorm, first_hurricane
        "len_rolling_sum": 7,
        "climo_nfilter": 41,
        "leadtimes": np.arange(0, 7*8, 7),
        "n_val_years": 6,
        "months_to_predict": (5, 6, 7, 8, 9, 10, 11), #(1,2,3,4,5,6, 7, 8, 9, 10, 11, 12),
        "predictors_only_in_season": False,
        "show_plots": False,
        # model parameters
        "rng_seed": 33,
        "warm_start": False,
        "class_weight": None,
        # logistic regression parameters
        "lr_tol": 1.e-6,
        "lr_max_iter": 1_000,
    },

    "exp001": {
        # experiment parameters
        "model_type": "random_forest",
        "predictand": "label_cyclogensis_boolean",
        "years": (1979, 2020),
        "storm_type": "first_tropicalstorm",  # first_tropicalstorm, first_hurricane
        "len_rolling_sum": 7,
        "climo_nfilter": 40,
        "leadtimes": np.arange(0, 7*10, 7),
        "n_val_years": 6, #num of years per fold
        "months_to_predict": (6, 7, 8, 9, 10, 11),
        "predictors_only_in_season": False,
        "show_plots": False,
        # model parameters
        "rng_seed": 42,
        "warm_start": False,
        "class_weight": None,
        # random forest parameters
        "rf_leaf_samples": 2,
        "rf_node_split": 4,
        "rf_number_of_trees": 500,
        "rf_tree_depth": 12,
        "rf_criterion": "entropy",
        "rf_max_feat": "sqrt",
        "rf_bootstrap": True,
        "rf_oob": True,
    },

    "exp002": {  # this should give you a repeat of logistic regression exp000!
        # experiment parameters
        "model_type": "mlp",
        "predictand": "label_cyclogensis_boolean",
        "years": (1979, 2020),
        "storm_type": "first_tropicalstorm",  # first_tropicalstorm, first_hurricane
        "len_rolling_sum": 7,
        "climo_nfilter": 40,
        "leadtimes": np.arange(0, 7*10, 7),
        "n_val_years": 6,
        "months_to_predict": (6, 7, 8, 9, 10, 11),
        "predictors_only_in_season": False,
        "show_plots": False,
        # model parameters
        "rng_seed": 33,
        "warm_start": False,
        "class_weight": None,
        # multi-layer perceptron parameters
        "mlp_tol": 1.e-6,
        "mlp_max_iter": 2_000,
        "mlp_learning_rate": "constant",
        "mlp_learning_rate_init": 0.001,
        "mlp_activation": "identity",
        "mlp_hidden_layer_sizes": [1, ],
        "mlp_alpha": 0.,
        "mlp_solver": "adam",
        "mlp_batch_size": 64,
    },
    
    "exp003": {
        # experiment parameters
        "model_type": "mlp",
        "predictand": "label_cyclogensis_boolean",
        "years": (1979, 2021),
        "storm_type": "first_tropicalstorm",  # first_tropicalstorm, first_hurricane
        "len_rolling_sum": 7,
        "climo_nfilter": 40,
        "leadtimes": np.arange(0, 7*10, 7),  # np.arange(0, 7*10, 7),
        "n_val_years": 6,
        "months_to_predict": (6, 7, 8, 9, 10, 11),
        "predictors_only_in_season": False,
        "show_plots": False,
        # model parameters
        "rng_seed": 33,
        "warm_start": False,
        "class_weight": None,
        # multi-layer perceptron parameters
        "mlp_tol": 1.e-6,
        "mlp_max_iter": 2_000, 
        "mlp_learning_rate": "constant",
        "mlp_learning_rate_init": 0.01,
        "mlp_activation": "identity",
        "mlp_hidden_layer_sizes": [2, ],
        "mlp_alpha": 0.,
        "mlp_solver": "adam",
        "mlp_batch_size": 64,
    },
    
    "exp004": {
        # experiment parameters
        "model_type": "mlp",
        "predictand": "label_cyclogensis_boolean",
        "years": (1979, 2021),
        "storm_type": "first_tropicalstorm",  # first_tropicalstorm, first_hurricane
        "len_rolling_sum": 7,
        "climo_nfilter": 40,
        "leadtimes": np.arange(0, 7*10, 7),  # np.arange(0, 7*10, 7),
        "n_val_years": 6,
        "months_to_predict": (6, 7, 8, 9, 10, 11),
        "predictors_only_in_season": False,
        "show_plots": False,
        # model parameters
        "rng_seed": 33,
        "warm_start": False,
        "class_weight": None,
        # multi-layer perceptron parameters
        "mlp_tol": 1.e-6,
        "mlp_max_iter": 2_000,
        "mlp_learning_rate": "constant",
        "mlp_learning_rate_init": 0.01,
        "mlp_activation": "identity",
        "mlp_hidden_layer_sizes": [3, ],
        "mlp_alpha": 0.,
        "mlp_solver": "adam",
        "mlp_batch_size": 64,
    },
}


def get_settings(experiment_name):
    settings = experiments[experiment_name]
    settings["exp_name"] = experiment_name

    return settings
