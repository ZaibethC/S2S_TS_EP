import numpy as np
import pandas as pd
from datetime import date
import plot_diagnostics
import scipy.signal as sig

import importlib as imp


def build_data(settings):
    # organize predictors and labels
    df_enso = get_enso_index(settings)
    df_mjo = get_mjo_index(settings)
    df_hurr = get_hurdat_data(settings)
    df_qbo = get_qbo_index(settings)
    df_uwind = get_uwind_data(settings)
    df_sst = get_sst_data(settings) 

    # put the data together, get climotology and define anomalies (may not need the anomalies though
    df_data = combine_data(settings, df_hurr, df_mjo, df_enso,df_qbo, df_uwind, df_sst)
    df_data = add_climatology(settings, df_data)

    return df_data


def get_enso_index(settings):
    table_enso = pd.read_csv('data/_nino34_anom_data.txt',
                             header=None,
                             delimiter='   ', #3 espacios
                             names=['year', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                                    'september', 'october', 'november', 'december'],
                             engine="python"
                             )

    df_enso = pd.DataFrame()
    for iyear in np.arange(0, table_enso.shape[0]):
        for imonth in np.arange(1, table_enso.shape[1]):
            df = pd.DataFrame(
                data={'year': [table_enso["year"][iyear]], 'month': [imonth], "enso": [table_enso.iloc[iyear, imonth]]})
            df_enso = pd.concat([df_enso, df], ignore_index=True)

    # grab years
    df_enso = df_enso[(df_enso["year"] >= settings["years"][0]) & (df_enso["year"] <= settings["years"][1])].reset_index(drop=True)

    return df_enso


def get_qbo_index(settings):
    table_qbo = pd.read_csv('data/anom_qbo.79toRealtime.csv',
                             header=None,
#                              delimiter='    ',
                             names=['year', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                                    'september', 'october', 'november', 'december'],
                             engine="python"
                             )

    df_qbo = pd.DataFrame()
    for iyear in np.arange(0, table_qbo.shape[0]):
        for imonth in np.arange(1, table_qbo.shape[1]):

            df = pd.DataFrame(
                data={'year': [table_qbo["year"][iyear]], 'month': [imonth], "qbo": [table_qbo.iloc[iyear, imonth]]})
            df_qbo = pd.concat([df_qbo, df], ignore_index=True)

    # grab years
    df_qbo = df_qbo[(df_qbo["year"] >= settings["years"][0]) & (df_qbo["year"] <= settings["years"][1])].reset_index(drop=True)

    return df_qbo


def get_mjo_index(settings):
    df_mjo = pd.read_csv('data/rmm_74toRealtime.csv',
#                          header=None,
                         delimiter=',',
#                          names=['year', 'month', 'day', 'rmm1', 'rmm2', 'phase', 'amplitude'],
                         engine="python",
                         )

    df_mjo = df_mjo[(df_mjo["year"] >= settings["years"][0]) & (df_mjo["year"] <= settings["years"][1])].reset_index(
        drop=True)

    # convert to integers
    df_mjo = df_mjo.astype({'year': 'int', 'month': 'int', 'day': 'int'})
    df_mjo = df_mjo.rename(columns={'RMM1':'rmm1','RMM2':'rmm2'})

    return df_mjo

def get_uwind_data(settings):
    df_uwind = pd.read_csv('data/uwind_daily_anomalies_1979to2021.csv',
                           engine="python"
                          ).iloc[:,1:]
    return df_uwind


def get_sst_data(settings):
    df_sst = pd.read_csv('data/sst_daily_anomalies_1979to2021.csv',
                           engine="python"
                          ).iloc[:,1:]
    return df_sst

def get_hurdat_data(settings):
    # read the data
    df_hurdat = pd.read_csv("data/HURDAT_49toRealtime21.csv",
#                             header=None,
#                             names=["date", "time", "count", "storm_type", "latitude", "longitude", "intensity"],
                            skipinitialspace=True,
                            ).iloc[:,1:]

    # grab the rows with storm information and label the storms
    df_hurr = pd.DataFrame()
    for index, row in df_hurdat.iterrows():
        if row["date"][:2] == "EP":
            # print(df_hurdat.iloc[index:index+int(row["count"])+1])
            df = df_hurdat.iloc[index + 1:index + int(row["count"]) + 1].copy()
            df["storm_name"] = row["date"]
            df_hurr = pd.concat([df_hurr, df], ignore_index=True)
    df_hurr = df_hurr.drop("count", axis=1)

    # add year, month and day to the dataframe
    year = np.zeros((df_hurr.shape[0],))
    month = np.zeros((df_hurr.shape[0],))
    day = np.zeros((df_hurr.shape[0],))
    for index, row in df_hurr.iterrows():
        year[index] = row["date"][:4]
        month[index] = row["date"][4:6]
        day[index] = row["date"][6:]

    df_hurr["year"] = year
    df_hurr["month"] = month
    df_hurr["day"] = day
    df_hurr = df_hurr[
        ['date', "year", "month", "day", "time", "storm_name", "storm_type", "latitude", "longitude", "intensity"]]

    # get years
    df_hurr = df_hurr[
        (df_hurr["year"] >= settings["years"][0]) & (df_hurr["year"] <= settings["years"][1])].reset_index(drop=True)

    # decide which records to grab; i.e. the first time the storm becomes hurricane strength
    if settings["storm_type"] == "first_hurricane":
#         df = df_hurr[df_hurr["storm_type"] == "HU"].copy()
        df = df_hurr[(df_hurr['intensity']>=64)].copy() 
        df_selection = pd.DataFrame()
        unique_storms = np.unique(df["storm_name"])
        for name in unique_storms:
            irows = np.where(df["storm_name"] == name)[0]
            ifirst = np.argmin(df["date"].iloc[irows])

            # grab ifirst:ifirst+1 so that it keeps the dataframe in rows, this is just a little trick
            df_selection = pd.concat([df_selection, df.iloc[irows[ifirst:ifirst + 1]]].copy(), ignore_index=True, )
        df_selection = df_selection.sort_values(by="date").reset_index(drop=True)

    elif settings["storm_type"] == "first_tropicalstorm":
#         df = df_hurr[(df_hurr["storm_type"] == "HU") | (df_hurr["storm_type"] == "TS")].copy()
        df = df_hurr[(df_hurr['intensity']>=34) | (df_hurr['intensity']>=64)].copy()
        df_selection = pd.DataFrame()
        unique_storms = np.unique(df["storm_name"])
        for name in unique_storms:
            irows = np.where(df["storm_name"] == name)[0]
            ifirst = np.argmin(df["date"].iloc[irows])

            # grab ifirst:ifirst+1 so that it keeps the dataframe in rows, this is just a little trick
            df_selection = pd.concat([df_selection, df.iloc[irows[ifirst:ifirst + 1]]].copy(), ignore_index=True, )
        df_selection = df_selection.sort_values(by="date").reset_index(drop=True)
    else:
        raise NotImplementedError("no such storm_type")

    df_hurr = df_selection.copy()
    
    #addition
    
    lat=np.empty(len(df_hurr['latitude']))
    lon=np.empty(len(df_hurr['longitude']))
    
    for ilat in range(0,len(df_hurr['latitude'])):
        if len(df_hurr['latitude'][ilat])==5:
            lat[ilat]=float(df_hurr['latitude'][ilat][:4])
        elif len(df_hurr['latitude'][ilat])==4:
            lat[ilat]=float(df_hurr['latitude'][ilat][:3])
        elif len(df_hurr['latitude'][ilat])>5 or len(df_hurr['latitude'][ilat])<4:
            print(df_hurr['latitude'][ilat])
        
        if (lat[ilat] < 5) or (lat[ilat]>25): #limiting the region that is being studied 
            lat[ilat] = np.nan 
        
    for ilon in range(0,len(df_hurr['longitude'])):
        if len(df_hurr['longitude'][ilon])==6:
    #         print((df_hurr['longitude'][ilon][:5]))
            lon[ilon]=float(df_hurr['longitude'][ilon][:5])
        elif len(df_hurr['longitude'][ilon])==5:
    #         print((df_hurr['longitude'][ilon][:4]))
            lon[ilon]=float(df_hurr['longitude'][ilon][:4])
        elif len(df_hurr['longitude'][ilon])<5 or len(df_hurr['longitude'][ilon])>6:
            print(df_hurr['longitude'][ilon][:3])
        
        if (lon[ilon] < 90) or (lon[ilon]>120): #limiting the region that is being studied 
            lon[ilon] = np.nan 
    
    df_selection['latitude'] = lat
    df_selection['longitude'] = lon
    
    df_hurr = df_selection.dropna().reset_index(drop=True)

    return df_hurr


def combine_data(settings, df_hurr, df_mjo, df_enso, df_qbo, df_uwind, df_sst):
    df_data = df_mjo.copy()
    df_data['u_data'] = df_uwind['u']
    df_data['sst_data'] = df_sst['sst']

    enso_vec = np.zeros((df_data.shape[0],))
    qbo_vec = np.zeros((df_data.shape[0],))
    nhurr_vec = np.zeros((df_data.shape[0],))
    diy_vec = np.zeros((df_data.shape[0],))
    uwind_vec = np.zeros((df_data.shape[0],))
    
    for index, row in df_data.iterrows():
        i = np.where((df_enso["year"] == int(row["year"])) & (df_enso["month"] == int(row["month"])))[0]
        enso_vec[index] = df_enso.iloc[i]["enso"].copy()
        
        i = np.where((df_qbo["year"] == int(row["year"])) & (df_qbo["month"] == int(row["month"])))[0]
        qbo_vec[index]  = df_qbo.iloc[i]['qbo'].copy()
        
#         i = np.where((df_uwind["year"] == int(row["year"])) & (df_uwind["month"] == int(row["month"])) & (
#                 df_uwind["day"] == int(row["day"])))[0]
#         uwind_vec[index]  = df_uwind.iloc[i]['u10'].copy()

        j = np.where((df_hurr["year"] == int(row["year"])) & (df_hurr["month"] == int(row["month"])) & (
                df_hurr["day"] == int(row["day"])))[0]
        nhurr_vec[index] = len(j)
        
#         j = np.where((df_uwind["year"] == int(row["year"])) & (df_uwind["month"] == int(row["month"])) & (
#                 df_uwind["day"] == int(row["day"])))[0]
#         uwind_vec[index] = len(j)

        # get day in year
        # to get around leap years, set Feb 29th to DIY of Feb 28
        if int(row["month"]) == 2 and int(row["day"]) == 29:
            # 1981 is not a leap year
            date_val = date(1981, 2, 28)
        else:
            date_val = date(1981, int(row["month"]), int(row["day"]))
        diy_vec[index] = date_val.toordinal() - date(date_val.year, 1, 1).toordinal() + 1

    df_data["diy"] = diy_vec
    df_data["date"] = df_data["year"] * 10_000 + df_data["month"] * 100 + df_data["day"] * 1
    df_data["enso_index"] = enso_vec
    df_data["qbo_index"]  = qbo_vec
    df_data["nhurr"] = nhurr_vec

    # compute forward rolling sum of hurricanes
    df_data["nhurr_roll"] = df_data.iloc[::-1]["nhurr"].rolling(settings["len_rolling_sum"]).sum().iloc[::-1]

    # compute cyclogensis boolean {0: no TCs formed, 1: at least one TC formed
    df_data["cyclogensis_boolean"] = np.heaviside(df_data["nhurr_roll"].values, 0.)

    # set day in year to integer
    df_data = df_data.astype({'diy': 'int', 'date': 'int'})

    return df_data


def compute_boolean_climatology(settings, df_data):
    # The climatology of cyclogenesis over a group of days (0=no, 1=yes)

    g = [1., 2., 1.]

    df_hurr_climo = df_data.groupby('diy', as_index=False)['cyclogensis_boolean'].mean()
    df_hurr_climo_shift = pd.DataFrame(np.roll(df_hurr_climo.values, -90),
                                       columns=df_hurr_climo.columns,
                                       )

    climo_smoothed = df_hurr_climo_shift["cyclogensis_boolean"]
    for f in range(0, settings["climo_nfilter"]):
        climo_smoothed = sig.filtfilt(g, np.sum(g), climo_smoothed)

    df_hurr_climo_final = pd.DataFrame()
    df_hurr_climo_final["diy"] = df_hurr_climo_shift["diy"].copy().reset_index(drop=True)
    df_hurr_climo_final["cyclogensis_raw_climo"] = df_hurr_climo_shift["cyclogensis_boolean"].copy().reset_index(drop=True)
    df_hurr_climo_final["cyclogensis_smooth_climo"] = climo_smoothed
    df_hurr_climo_final = df_hurr_climo_final.reset_index(drop=True)
    df_hurr_climo_final = df_hurr_climo_final.sort_values(by="diy").reset_index(drop=True)

    # add reference frequency
    df_hurr_climo_final["cyclogensis_reference_frequency"] = df_data[df_data['month'].isin(settings["months_to_predict"])][
        "cyclogensis_boolean"].mean()

    # if settings["show_plots"]:
    plot_diagnostics.plot_climatology(settings, df_hurr_climo_final)

    return df_hurr_climo_final


def compute_climatology(settings, df_data):
    # compute the climatology and anomalies of storms for later use
    # this is NOT what Slade and Maloney predicted, see compute_boolean_climatology()
    g = [1., 2., 1.]

    df_hurr_climo = df_data.groupby('diy', as_index=False)['nhurr_roll'].mean()
    df_hurr_climo_shift = pd.DataFrame(np.roll(df_hurr_climo.values, -90),
                                       columns=df_hurr_climo.columns,
                                       )

    climo_smoothed = df_hurr_climo_shift["nhurr_roll"]
    for f in range(0, settings["climo_nfilter"]):
        climo_smoothed = sig.filtfilt(g, np.sum(g), climo_smoothed)

    df_hurr_climo_final = pd.DataFrame()
    df_hurr_climo_final["diy"] = df_hurr_climo_shift["diy"].copy().reset_index(drop=True)
    df_hurr_climo_final["nhurr_roll_raw_climo"] = df_hurr_climo_shift["nhurr_roll"].copy().reset_index(drop=True)
    df_hurr_climo_final["nhurr_roll_smooth_climo"] = climo_smoothed
    df_hurr_climo_final = df_hurr_climo_final.reset_index(drop=True)
    df_hurr_climo_final = df_hurr_climo_final.sort_values(by="diy").reset_index(drop=True)

    # add reference frequency
    df_hurr_climo_final["nhurr_roll_reference_frequency"] = df_data[df_data['month'].isin(
        settings["months_to_predict"])]["nhurr_roll"].mean()

    return df_hurr_climo_final


def add_climatology(settings, df_data):

    # add climatology and anomalies of the cyclogensis boolean (0,1)
    df_hurr_climo_final = compute_boolean_climatology(settings, df_data)

    # add climatology and anomalies
    smooth_climo_vec = np.zeros((df_data.shape[0],))
    raw_climo_vec = np.zeros((df_data.shape[0],))
    for index, row in df_data.iterrows():
        j = np.where(df_hurr_climo_final["diy"] == int(row["diy"]))[0]
        smooth_climo_vec[index] = df_hurr_climo_final.iloc[j]["cyclogensis_smooth_climo"]
        raw_climo_vec[index] = df_hurr_climo_final.iloc[j]["cyclogensis_raw_climo"]

    df_data["cyclogensis_smooth_climo"] = smooth_climo_vec
    df_data["cyclogensis_raw_climo"] = raw_climo_vec
    df_data["cyclogensis_anom"] = df_data["cyclogensis_boolean"] - df_data["cyclogensis_smooth_climo"]
    df_data["cyclogensis_reference_frequency"] = np.ones((df_data.shape[0],)) * \
                                                 np.unique(df_hurr_climo_final["cyclogensis_reference_frequency"])

    # add climatology and anomalies of the total counts of storms
    df_hurr_climo_final = compute_climatology(settings, df_data)

    # add climatology and anomalies
    smooth_climo_vec = np.zeros((df_data.shape[0],))
    raw_climo_vec = np.zeros((df_data.shape[0],))
    anom_vec = np.zeros((df_data.shape[0],))

    for index, row in df_data.iterrows():
        j = np.where(df_hurr_climo_final["diy"] == int(row["diy"]))[0]
        smooth_climo_vec[index] = df_hurr_climo_final.iloc[j]["nhurr_roll_smooth_climo"]
        raw_climo_vec[index] = df_hurr_climo_final.iloc[j]["nhurr_roll_raw_climo"]
        anom_vec[index] = row["nhurr_roll"] - smooth_climo_vec[index]

    df_data["nhurr_roll_smooth_climo"] = smooth_climo_vec
    df_data["nhurr_roll_raw_climo"] = raw_climo_vec
    df_data["nhurr_roll_anom"] = df_data["nhurr_roll"] - df_data["nhurr_roll_smooth_climo"]
    df_data["nhurr_roll_reference_frequency"] = np.ones((df_data.shape[0],)) * \
                                                 np.unique(df_hurr_climo_final["nhurr_roll_reference_frequency"])
    
    hurr_count = np.zeros([max(df_data['diy'])])

    for d in np.arange(0,max(df_data['diy'])):
        hurr_count[d] = sum((df_data["cyclogensis_boolean"].where((df_data['diy'] == d))).dropna())
    
    bool_hurr = np.zeros([len(df_data)])

    for d in range(0,len(df_data)):
        for y in range(1,len(hurr_count)):
            if df_data['diy'][d] == y: 
                bool_hurr[d] = hurr_count[y-1]
                
    df_data['hurr_counts'] = bool_hurr
    
    
    #smoothing the hurricane counts 
    g = [1., 2., 1.]

    df_hurr_climo = df_data.groupby('diy', as_index=False)['hurr_counts'].mean()
    df_hurr_climo_shift = pd.DataFrame(np.roll(df_hurr_climo.values, 0),
                                       columns=df_hurr_climo.columns,
                                       )

    climo_smoothed = df_hurr_climo_shift["hurr_counts"]
#     climo_smoothed = bool_hurr
    
    for f in range(0, settings["climo_nfilter"]):
        climo_smoothed = sig.filtfilt(g, np.sum(g), climo_smoothed)
    
    df_hurr_climo_final["nhurr_roll_smooth_counts"] = climo_smoothed
        
    smooth_climo_vec = np.zeros((df_data.shape[0],))
        
    for index, row in df_data.iterrows():
        j = np.where(df_hurr_climo_final["diy"] == int(row["diy"]))[0]
        smooth_climo_vec[index] = df_hurr_climo_final.iloc[j]["nhurr_roll_smooth_counts"]
        
    df_data['hurr_counts_smoothed'] = smooth_climo_vec

    return df_data
