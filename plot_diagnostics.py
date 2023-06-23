import matplotlib.pyplot as plt
import numpy as np
import save_load_model

# figure commands
dpiFig = 400
plt.style.use("seaborn-white")
plt.rc('font', size=16)
plt.rc('savefig', facecolor='white')
plt.rc('axes', facecolor='white')


def plot_labels_check(settings, df_data_model):

    plt.figure(figsize=(18, 9))

    # shift the data up so that we can see both sets of curves at the same time
    plt.plot(3+df_data_model["nhurr_roll"], '-b', alpha=.5, label='nhurr_roll')
    plt.plot(3+df_data_model["cyclogensis_boolean"], '--r', alpha=.5, label='cyclogensis_boolean')

    plt.plot(df_data_model["label_nhurr_roll"], '-b', label='label_nhurr_roll')
    plt.plot(df_data_model["label_cyclogensis_boolean"], '--r', label='label_cyclogensis_boolean')

    plt.xlim(200, 500)
    plt.legend()
    plt.show()


def plot_climatology(settings, df_hurr_climo_final):
    plt.figure(figsize=(10,6))
    plt.plot(df_hurr_climo_final["diy"], df_hurr_climo_final["cyclogensis_raw_climo"]*100)
    plt.plot(df_hurr_climo_final["diy"], df_hurr_climo_final["cyclogensis_smooth_climo"]*100, color="orange", linewidth=3)

    plt.ylim(0,100)
    plt.xlabel('day of year')
    plt.ylabel('% probability')
    plt.title('Climatology of cyclogenesis probability (' + str(settings["len_rolling_sum"]) + ' day window)')
    plt.show()


def plot_metrics(settings, df_metrics, show_plots=True):
    # define colors
    colors = ("#6c757d","#009969","#8b0053","#ff9c23","#00639a","#9359ba","#a0b2cd","#96bd52","#847149","#c45245")

    for metric_to_plot in ("precision", "recall", "f1_score", "accuracy", 'brier_score', 'brier_skill_score',
                           'brier_skill_score_baseline_diff', 'accuracy_baseline_diff', 'f1_score_baseline_diff'):

        plt.figure(figsize=(15, 8))
        for ilead, leadtime in enumerate(df_metrics["leadtime"].unique()):
            df_plot = df_metrics[df_metrics["leadtime"] == leadtime].copy().reset_index(drop=True)

            for ifeat, features_type in enumerate(df_plot["features_type"].unique()):
                # for iter, features_type in enumerate(("baseline","mjo","all_mjo")):
                if features_type == "baseline":
                    if ilead == 0:
                        min_baseline = df_plot[df_plot["features_type"] == features_type][metric_to_plot].min()
                        max_baseline = df_plot[df_plot["features_type"] == features_type][metric_to_plot].max()
                        plt.fill_between(np.arange(-90, 90), min_baseline, max_baseline, color=colors[ifeat],
                                         alpha=.1)
                        yplot = df_metrics[(df_metrics.features_type == features_type)].groupby(
                            by=["leadtime"])[metric_to_plot].median()
                        plt.plot(yplot.index, yplot, '-', color=colors[ifeat], linewidth=3, alpha=.75,
                                 label=features_type,)

                else:
#                     if (ilead == 0)  and (features_type in ("all_rmm", "rmm")):
#                         yplot = df_metrics[(df_metrics.features_type == features_type)].groupby(
#                             by=["leadtime"])[metric_to_plot].median()
#                         plt.plot(yplot.index+(ifeat-len(df_plot["features_type"].unique())/2)/4., yplot, # linestyle='-',
#                                  color=colors[ifeat], linewidth=3, alpha=.75,)

                    plt.plot(
                        np.ones((settings["n_folds"],)) * leadtime + (
                                ifeat - len(df_plot["features_type"].unique()) / 2) / 4.,
                        df_plot[df_plot["features_type"] == features_type][metric_to_plot],
                        '.',
                        markersize=7,
                        alpha=.2,
                        label=None,
                        color=colors[ifeat],
                    )
                    plt.plot(
                        leadtime + (ifeat - len(df_plot["features_type"].unique()) / 2) / 4.,
                        df_plot[df_plot["features_type"] == features_type][metric_to_plot].median(),
                        'o',
                        markersize=9,
                        alpha=.95,
                        markeredgecolor="gray",
                        label=features_type,
                        color=colors[ifeat],
                    )

        plt.grid(True)
        if metric_to_plot == "brier_score":
            plt.ylim(0, .3)
        elif metric_to_plot == "brier_skill_score":
            plt.ylim(-.05, .28)
            plt.axhline(y=0, color="black", linewidth=2)
        elif metric_to_plot == "brier_skill_score_baseline_diff" or \
                metric_to_plot == "accuracy_baseline_diff" or metric_to_plot == "f1_score_baseline_diff":
            plt.ylim(-.08, .08)
            plt.axhline(y=0, color="black", linewidth=2)
        elif metric_to_plot == "accuracy":
            plt.ylim(.5, .85)
        else:
            plt.ylim(0, 1.)
        plt.xlim(df_metrics["leadtime"].unique().min() - 5, df_metrics["leadtime"].unique().max() + 5)
        plt.xlabel("leadtime (days)")
        plt.xticks(df_metrics["leadtime"].unique(), df_metrics["leadtime"].unique())
        hand, leg = plt.gca().get_legend_handles_labels()
        plt.legend(hand[:ifeat + 1], leg[:ifeat + 1],
                   facecolor='white', framealpha=1, frameon=True, fontsize=12)
        plt.title(metric_to_plot + ' [' + settings["exp_name"] + ']', fontsize=24)

        figure_savename = save_load_model.get_exp_filename(settings["exp_name"])
        plt.savefig("figures/"  + figure_savename +'_'+ features_type + "_metrics_vs_leadtime_" + metric_to_plot + '79to21_counts.png',dpi=600, bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
