import os
import glob
from pylab import *

# import colors
from palettable.colorbrewer.qualitative import Set2_7

import json
import numpy as np
import matplotlib.pyplot as plt

colors = Set2_7.mpl_colors

params = {
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.figsize": [2.5, 4.5],
}
rcParams.update(params)

name = "chimney"
main_path = "/home/itsikelis/Documents/PhD/Code/se3_trajopt/src/examples/cem_exps/results/" + name

def load(directory):
    f_list = glob.glob(directory + "*/*.json")
    cost_histories = []
    wall_histories = []
    for file in f_list:
        with open(file, 'r') as f:
            data = json.load(f)
            elite_costs = data["solution"]["elite_cost_history"]
            wall_time = data["solution"]["wall_time_sec"]
            cost_histories.append(elite_costs)
            wall_histories.append(wall_time)

    cost_histories = np.array(cost_histories)
    wall_histories = np.array(wall_histories)
    return cost_histories, wall_histories

chimney_low, wall_low = load(main_path + "/talos/climb_low/")
chimney_high, wall_high = load(main_path + "/talos/climb_high/")

# talos_costs = -talos_costs
# print(talos_costs)

# fig = figure()
# ax = fig.add_subplot(111)
#
# # ax.set_yscale('log')  # Set y-axis to log scale
#
# ## Plot costs ##
# xx = np.arange(0, talos_costs.shape[-1], 1)
# ax.plot(xx, np.median(talos_costs, axis=0), label="cost")
# low = 25
# up = 75
# plow = np.percentile(talos_costs, low, axis=0)
# phigh = np.percentile(talos_costs, up, axis=0)
# ax.fill_between(xx, plow, phigh, alpha=0.25, linewidth=0)#, color='#B22400')
#
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()
# ax.tick_params(axis="x", direction="out")
# ax.tick_params(axis="y", length=0)
#
# ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1)
# ax.set_axisbelow(True)
#
# # ax.set_xticklabels(nice_names)
#
# ax.set_xlabel("Iterations", fontsize=12)
# ax.set_ylabel("Objective Value", fontsize=12)
# ax.set_title("Time Minimization", fontsize=14)
#
# ax.set_ylim([-5, 0])
#
# ax.legend(loc=8)
#
# fig.tight_layout()
#
# # savefig(name + "costs" + ".pdf")
# # close()
