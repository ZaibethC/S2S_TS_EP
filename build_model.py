import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def make_rf_classifier(settings):
    model = RandomForestClassifier(n_estimators=settings["rf_number_of_trees"], random_state=settings["rng_seed"],
                                min_samples_split=settings["rf_node_split"],
                                min_samples_leaf=settings["rf_leaf_samples"], criterion=settings["rf_criterion"],
                                max_depth=settings["rf_tree_depth"],
                                max_features=settings["rf_max_feat"], bootstrap=settings["rf_bootstrap"],
                                warm_start=settings["warm_start"],
                                class_weight=settings["class_weight"],
                                oob_score=settings["rf_oob"],
                                )
    return model


def make_rf_regressor(settings):
    model = RandomForestRegressor(n_estimators=settings["rf_number_of_trees"], random_state=settings["rng_seed"],
                                min_samples_split=settings["rf_node_split"],
                                min_samples_leaf=settings["rf_leaf_samples"], criterion=settings["rf_criterion"],
                                max_depth=settings["rf_tree_depth"],
                                max_features=settings["rf_max_feat"], bootstrap=settings["rf_boots"],
                                warm_start=settings["warm_start"],
                                oob_score=settings["rf_oob"],
                                )
    return model


def make_lr_classifier(settings):
    model = LogisticRegression(warm_start=settings["warm_start"],
                            class_weight=settings["class_weight"],
                            random_state=settings["rng_seed"],
                            tol=settings["lr_tol"],
                            max_iter=settings["lr_max_iter"],
                            # penalty='l1',
                            )

    return model


def make_mlp_classifier(settings):
    model = MLPClassifier(hidden_layer_sizes=settings["mlp_hidden_layer_sizes"],
                          activation=settings["mlp_activation"],
                          learning_rate=settings["mlp_learning_rate"],
                          learning_rate_init=settings["mlp_learning_rate_init"],
                          max_iter=settings["mlp_max_iter"],
                          random_state=settings["rng_seed"],
                          tol=settings["mlp_tol"],
                          warm_start=settings["warm_start"],
                          alpha=settings["mlp_alpha"],
                          solver=settings["mlp_solver"],
                          batch_size=settings["mlp_batch_size"],
                          verbose=False,
                          )

    return model
