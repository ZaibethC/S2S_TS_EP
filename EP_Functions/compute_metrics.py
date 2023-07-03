import numpy as np
import sklearn


def declare_empty_metrics(n_folds):
    metrics_dict = {
        "iterfold": np.zeros((n_folds,)),
        "accuracy": np.zeros((n_folds,)),
        "matthews_corrcoef": np.zeros((n_folds,)),
        "f1_score": np.zeros((n_folds,)),
        "precision": np.zeros((n_folds,)),
        "recall": np.zeros((n_folds,)),
        "avg_precision_score": np.zeros((n_folds,)),
        "brier_score": np.zeros((n_folds,)),
        "brier_skill_score": np.zeros((n_folds,)),
    }

    return metrics_dict


def compute_metrics(metrics_dict, iterfold, model, x_train, y_train, x_val, y_val):

    metrics_dict["iterfold"][iterfold] = iterfold
    metrics_dict["accuracy"][iterfold] = sklearn.metrics.accuracy_score(y_val, model.predict(x_val))
    metrics_dict["matthews_corrcoef"][iterfold] = sklearn.metrics.matthews_corrcoef(y_val, model.predict(x_val))
    metrics_dict["f1_score"][iterfold] = sklearn.metrics.f1_score(y_val, model.predict(x_val))
    metrics_dict["precision"][iterfold] = sklearn.metrics.precision_score(y_val, model.predict(x_val), zero_division=0)
    metrics_dict["recall"][iterfold] = sklearn.metrics.recall_score(y_val, model.predict(x_val))
    metrics_dict["avg_precision_score"][iterfold] = sklearn.metrics.average_precision_score(y_val, model.predict(x_val))
    metrics_dict["brier_score"][iterfold] = sklearn.metrics.brier_score_loss(y_val, model.predict_proba(x_val)[:, 1])

    # compute Brier Skill Score
    bs_ref = sklearn.metrics.brier_score_loss(y_val, np.ones(np.shape(y_val))*np.mean(y_train))
    metrics_dict["brier_skill_score"][iterfold] = 1. - metrics_dict["brier_score"][iterfold] / bs_ref

    return metrics_dict


def add_difference_metrics(df_rf_results):

    bss_diff_vec = np.zeros((df_rf_results.shape[0],))
    acc_diff_vec = np.zeros((df_rf_results.shape[0],))
    f1_diff_vec = np.zeros((df_rf_results.shape[0],))

    for index, row in df_rf_results.iterrows():

        # brier_skill_score difference
        base_value = df_rf_results[(df_rf_results["features_type"] == "baseline") &
                                   (df_rf_results["iterfold"] == row["iterfold"]) &
                                   (df_rf_results["leadtime"] == row["leadtime"])
                                   ][["brier_skill_score", "accuracy", "f1_score"]]

        bss_diff_vec[index] = row["brier_skill_score"] - base_value["brier_skill_score"].values[0]
        acc_diff_vec[index] = row["accuracy"] - base_value["accuracy"].values[0]
        f1_diff_vec[index] = row["f1_score"] - base_value["f1_score"].values[0]

    df_rf_results["brier_skill_score_baseline_diff"] = bss_diff_vec
    df_rf_results["accuracy_baseline_diff"] = acc_diff_vec
    df_rf_results["f1_score_baseline_diff"] = f1_diff_vec

    return df_rf_results