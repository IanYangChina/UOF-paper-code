import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dcp


def smoothed_plot(file, data, x_label="Timesteps", y_label="Success rate", window=5):
    N = len(data)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(data[max(0, t - window):(t + 1)])
    x = [i for i in range(N)]
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if x_label == "Epoch":
        x_tick_interval = len(data) // 10
        plt.xticks([n * x_tick_interval for n in range(11)])
    plt.plot(x, running_avg)
    plt.savefig(file, bbox_inches='tight', dpi=500)
    plt.close()


# Plot multiple tendency lines in 1 figure
def smoothed_plot_multi_line(file, data,
                             legend=None, legend_loc="upper right",
                             x_label='Timesteps', y_label="Success rate", window=5):
    plt.ioff()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if x_label == "Epoch":
        x_tick_interval = len(data[0]) // 10
        plt.xticks([n * x_tick_interval for n in range(11)])

    for t in range(len(data)):
        N = len(data[t])
        x = [i for i in range(N)]
        if window != 0:
            running_avg = np.empty(N)
            for n in range(N):
                running_avg[n] = np.mean(data[t][max(0, n - window):(n + 1)])
        else:
            running_avg = data[t]

        plt.plot(x, running_avg)

    if legend is None:
        legend = [str(n) for n in range(len(data))]
    plt.legend(legend, loc=legend_loc)
    plt.savefig(file, bbox_inches='tight', dpi=500)
    plt.close()


def smoothed_plot_mean_deviation(file, data, color=None, legend=None, legend_inside=False, legend_loc="lower left", ncol=None,
                                 x_label='Timesteps', y_label="Success rate", window=5):
    plt.ioff()
    if color is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                  'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    elif len(color) < len(data):
        raise ValueError("not enough colors")
    else:
        colors = color
    plt.ylabel(y_label, fontsize=15)
    plt.yticks(fontsize=14)
    plt.xlabel(x_label, fontsize=15)
    if x_label == "Epoch":
        x_tick_interval = len(data[0]["mean"]) // 10
        plt.xticks([n * x_tick_interval for n in range(11)], fontsize=14)

    N = len(data[0]["mean"])
    x = [i for i in range(N)]
    for i in range(len(data)):
        case_data = data[i]
        for key in case_data:
            running_avg = np.empty(N)
            for n in range(N):
                running_avg[n] = np.mean(case_data[key][max(0, n - window):(n + 1)])

            case_data[key] = dcp(running_avg)

        plt.fill_between(x, case_data["upper"], case_data["lower"], alpha=0.3, color=colors[i], label='_nolegend_')
        plt.plot(x, case_data["mean"], color=colors[i])

    if legend is None:
        legend = [str(n) for n in range(len(data))]
    if ncol is None:
        ncol = len(legend)
    if not legend_inside:
        plt.legend(legend, fontsize=15, bbox_to_anchor=(0., 1.02, 1., .102), loc=legend_loc, ncol=ncol, mode="expand", borderaxespad=0.)
    else:
        plt.legend(legend, fontsize=15, loc=legend_loc)
    plt.savefig(file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
