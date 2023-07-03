import numpy as np
import pandas as pd
import plot_diagnostics


def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def make_model_data(settings, df_data):
    # make label data shifted forward by leadtime
    # Since elements that roll beyond the last position are re-introduced at the first using np.roll,
    # you may want to set them to nan

    df = df_data[["diy", 'year', "month", 'day', "nhurr", "nhurr_roll", "cyclogensis_smooth_climo",
                  "cyclogensis_raw_climo", "cyclogensis_boolean", "cyclogensis_anom", 'hurr_counts','hurr_counts_smoothed',]].copy()
    rolled_values = np.roll(df.values, -settings["leadtime"], axis=0)
    if settings["leadtime"] > 0:
        rolled_values[-settings["leadtime"]:, -1] = np.nan  # if leadtime is zero the entire thing goes to nan
    df_shift = pd.DataFrame(rolled_values,
                            columns=["label_diy", 'label_year', "label_month", 'label_day', "label_nhurr",
                                     "label_nhurr_roll", "label_cyclogensis_smooth_climo",
                                     "label_cyclogensis_raw_climo", "label_cyclogensis_boolean",
                                     "label_cyclogensis_anom",'label_hurr_counts', 'label_hurr_counts_smoothed'],)
    df_data_model = pd.concat([df_data, df_shift], axis=1).reset_index(drop=True)

    # create targets, 0 if no hurricanes, 1 if there is one or more
    df_data_model["leadtime"] = settings["leadtime"]


    
    
    # add predictor of last week's MJO
    df = df_data_model[["rmm1", "rmm2", "phase", "amplitude"]]
    rolled_values = np.roll(df.values, 7, axis=0)
    rolled_values[:7, -1] = np.nan
    df_shift = pd.DataFrame(rolled_values,
                            columns=["rmm1-7", "rmm2-7", "phase-7", "amplitude-7"],
                            )
    df_data_model = pd.concat([df_data_model, df_shift], axis=1).reset_index(drop=True)
    

    # drop the nans from the np.roll
    
    df_data_model = df_data_model.dropna(axis=0).reset_index(drop=True)

    # predict for the hurricane season only
    df_data_model = df_data_model[df_data_model['label_month'].isin(settings["months_to_predict"])].reset_index(
        drop=True)
    if settings["predictors_only_in_season"]:
        # predict samples for when predictors are ALSO only in the hurricane season
        df_data_model = df_data_model[df_data_model['month'].isin(settings["months_to_predict"])].reset_index(drop=True)

    if settings["show_plots"]:
        plot_diagnostics.plot_labels_check(settings, df_data_model)

    # split into training + validation
    df_data_train = df_data_model[~df_data_model["year"].isin(settings["val_years"])]
    df_data_val = df_data_model[df_data_model["year"].isin(settings["val_years"])]

    return df_data_model, df_data_train, df_data_val


def get_features_labels(settings, features_type, df_data_train, df_data_val):
    if features_type == "reference_frequency":
        features = ["cyclogensis_reference_frequency"]
    elif features_type == "baseline":
        features = ["label_hurr_counts_smoothed"]
    elif features_type == "mjo":
        features = ["label_hurr_counts", "amplitude", "phase"]
    elif features_type == "rmm":
        features = ["label_hurr_counts", "rmm1", "rmm2"]
    elif features_type == "enso":
        features = ["label_hurr_counts_smoothed", "enso_index"]
    elif features_type == "all_mjo":
        features = ["label_hurr_counts_smoothed", "enso_index", "amplitude", "phase"]
    elif features_type == "all_rmm":
        features = ["label_hurr_counts_smoothed", "enso_index", "rmm1", "rmm2"]
    elif features_type == "mjo_mjo-7":
        features = ["label_hurr_counts_smoothed", "amplitude", "phase", "amplitude-7", "phase-7"]
    elif features_type == "all_mjo_mjo-7":
        features = ["label_hurr_counts_smoothed", "enso_index", "amplitude", "phase", "amplitude-7", "phase-7"]
    elif features_type == "all_rmm_rmm-7":
        features = ["label_hurr_counts_smoothed", "enso_index", "rmm1", "rmm2", "rmm1-7", "rmm2-7"]
    elif features_type == "all_rmm_rmm-14":
        features = ["label_hurr_counts_smoothed", "enso_index", "rmm1", "rmm2","rmm1-14", "rmm2-14"]
    elif features_type == "rmm_rmm-7":
        features = ["label_hurr_counts_smoothed", "rmm1", "rmm2", "rmm1-7", "rmm2-7"]        
    
    elif features_type == "all_rmm_rmm-7_uwind":
        features = ["label_hurr_counts_smoothed", "enso_index", "rmm1", "rmm2", "rmm1-7", "rmm2-7","u_data"]
    
    elif features_type == "uwind":
        features = ["u_data"]
    
    elif features_type == "all_uwind":
        features = ["label_hurr_counts_smoothed","enso_index","u_data"]
        
    elif features_type == "all_sst":
        features = ["label_hurr_counts_smoothed","enso_index","sst_data"]        
        
    elif features_type == "all_uwind_sst":
        features = ["label_hurr_counts_smoothed","enso_index","u_data","sst_data"]
    
    elif features_type == "all_rmm_rmm-7_uwind_sst":  # ESTE 
        features = ["label_hurr_counts_smoothed", "enso_index", "rmm1", "rmm2", "rmm1-7", "rmm2-7","u_data", "sst_data"]
        
    elif features_type == "all_rmm_rmm-7_uwind_sst_diy":
        features = ["label_hurr_counts_smoothed", "enso_index", "rmm1", "rmm2", "rmm1-7", "rmm2-7","u_data", "sst_data","diy"]
        
    elif features_type == "all_rmm_rmm-7_sst":
        features = ["label_hurr_counts_smoothed", "enso_index", "rmm1", "rmm2", "rmm1-7", "rmm2-7","sst_data"]
        
    else:
        
        raise NotImplementedError("no such fit_type")

    # define training and validation sets
    if len(features) == 1:
        x_train = df_data_train[features].values.reshape(-1, 1)
        x_val = df_data_val[features].values.reshape(-1, 1)
    else:
        x_train = df_data_train[features].values
        x_val = df_data_val[features].values

    y_train = df_data_train[settings["predictand"]]
    y_val = df_data_val[settings["predictand"]]

    return x_train, y_train, x_val, y_val


def get_val_years(settings, df_data):
    rng = np.random.default_rng(settings["rng_seed"])
    shuffled_years = df_data["year"].unique()
    
    # Choosing which year will be part of the testing data
    test_year = 1997
    
#     shuffled_years = np.where(shuffled_years_all != test_year, shuffled_years_all, np.nan) # Year to forecast 
    
#     shuffled_years = shuffled_years[~np.isnan(shuffled_years)]
    
    rng.shuffle(shuffled_years)
    val_year_split = np.array_split(shuffled_years, len(shuffled_years) / settings["n_val_years"])

    return val_year_split



def make_prueba_data(settings, df_data):
    # make label data shifted forward by leadtime
    # Since elements that roll beyond the last position are re-introduced at the first using np.roll,
    # you may want to set them to nan

    df = df_data[["diy", 'year', "month", 'day', "nhurr", "nhurr_roll", "cyclogensis_smooth_climo",
                  "cyclogensis_raw_climo", "cyclogensis_boolean", "cyclogensis_anom", 'hurr_counts',]].copy()
    rolled_values = np.roll(df.values, -settings["leadtime"], axis=0)
    if settings["leadtime"] > 0:
        rolled_values[-settings["leadtime"]:, -1] = np.nan  # if leadtime is zero the entire thing goes to nan
    df_shift = pd.DataFrame(rolled_values,
                            columns=["label_diy", 'label_year', "label_month", 'label_day', "label_nhurr",
                                     "label_nhurr_roll", "label_cyclogensis_smooth_climo",
                                     "label_cyclogensis_raw_climo", "label_cyclogensis_boolean",
                                     "label_cyclogensis_anom",'label_hurr_counts'],)
    df_data_model = pd.concat([df_data, df_shift], axis=1).reset_index(drop=True)

    # create targets, 0 if no hurricanes, 1 if there is one or more
    df_data_model["leadtime"] = settings["leadtime"]


    
    
    # add predictor of last week's MJO
    df = df_data_model[["rmm1", "rmm2", "phase", "amplitude"]]
    rolled_values = np.roll(df.values, 7, axis=0)
    rolled_values[:7, -1] = np.nan
    df_shift = pd.DataFrame(rolled_values,
                            columns=["rmm1-7", "rmm2-7", "phase-7", "amplitude-7"],
                            )
    df_data_model = pd.concat([df_data_model, df_shift], axis=1).reset_index(drop=True)
    

    # drop the nans from the np.roll
    
    df_data_model = df_data_model.dropna(axis=0).reset_index(drop=True)

    # predict for the hurricane season only
    df_data_model = df_data_model[df_data_model['label_month'].isin(settings["months_to_predict"])].reset_index(
        drop=True)
    if settings["predictors_only_in_season"]:
        # predict samples for when predictors are ALSO only in the hurricane season
        df_data_model = df_data_model[df_data_model['month'].isin(settings["months_to_predict"])].reset_index(drop=True)



    return df_data_model


def get_features_prueba(settings, features_type, df_data_model_prueba):
    if features_type == "all_rmm_rmm-7_uwind_sst":  # ESTE 
        features = ["label_hurr_counts", "enso_index", "rmm1", "rmm2", "rmm1-7", "rmm2-7","u_data", "sst_data"]        
    else:
        
        raise NotImplementedError("no such fit_type")
        
    test_year = df_data_model_prueba[features].values

    return test_year




#%%
