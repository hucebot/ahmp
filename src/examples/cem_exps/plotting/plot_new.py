import os
import glob

import json
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import *

main_path = (
    "/home/itsikelis/Workspaces/se3_trajopt/src/examples/cem_exps/plotting/results/"
)


def load(directory):
    f_list = glob.glob(directory + "*/*.json")
    cost_histories = []
    wall_histories = []
    for file in f_list:
        with open(file, "r") as f:
            data = json.load(f)
            elite_costs = data["solution"]["elite_cost_history"]
            wall_time = data["solution"]["wall_time_sec"]
            cost_histories.append(elite_costs)
            wall_histories.append(wall_time)

    cost_histories = -np.array(cost_histories) / 10
    wall_histories = np.array(wall_histories)
    return cost_histories, wall_histories


abl_30_cost, abl_30_wall = load(main_path + "ablation_study/ablate_30/")
abl_50_cost, abl_50_wall = load(main_path + "ablation_study/ablate_50/")
abl_80_cost, abl_80_wall = load(main_path + "ablation_study/ablate_80/")


# Create figure
fig = create_fig()
ax1 = fig.add_subplot(111)  # , facecolor="#DAEAF2")
# ax1 = fig.add_subplot(131)  # , facecolor="#DAEAF2")
# ax2 = fig.add_subplot(132)
# ax3 = fig.add_subplot(133)
adjust_subplots(fig)

# plot_median_and_percentiles(ax1, abl_30_cost, 0, "30%", True)
# plot_median_and_percentiles(ax2, abl_50_cost, 1, "50%", True)
# plot_median_and_percentiles(ax3, abl_80_cost, 2, "80%", True)
# plot_line(ax1, abl_30_cost[0], 0, "30%", True)
# plot_line(ax1, abl_50_cost[0], 1, "50%", True)
# plot_line(ax1, abl_80_cost[0], 2, "80%", True)

# xticks = np.arange(0, 4, 1)
# xlabels = np.arange(1, 5, 1)
# yticks = np.linspace(0, 3000, 5)
# ylabels = np.linspace(0, 3000, 5)

# add_x_ticks(ax1, xticks, xlabels)
# add_y_ticks(ax1, yticks, ylabels)
# add_x_ticks(ax2, xticks, xlabels)
# add_y_ticks(ax2, yticks, ylabels)
# add_x_ticks(ax3, xticks, xlabels)
# add_y_ticks(ax3, yticks, ylabels)

# add_legend(ax2)
# add_legend(ax3)

bp_data = [
    abl_30_wall,
    abl_50_wall,
    abl_80_wall,
]
bp_labels = ["30%", "50%", "80%"]

plot_boxplots(ax1, bp_data, bp_labels)

yticks = np.linspace(0, 900, 5)
xticks = []
add_x_ticks(ax1, xticks)
add_y_ticks(ax1, yticks)

main_title = "Ablation Study"
sec_title = "Violated constraints after each iteration per percentage of elites"

add_main_title(ax1, main_title)
add_secondary_title(ax1, sec_title)

plt.show()
# fig.savefig("ablation_new.pdf")
