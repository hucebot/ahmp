import os
import glob
from pylab import *

# import colors
from palettable.colorbrewer.qualitative import Set2_7

import json
import numpy as np
import matplotlib.pyplot as plt

main_path = "/home/itsikelis/Documents/PhD/Code/se3_trajopt/src/examples/cem_exps/plotting/results/"

def load(directory):
    f_list = glob.glob(directory + "*/*.json")
    cost_histories = []
    wall_histories = []
    for file in f_list:
        with open(file, 'r') as f:
            data = json.load(f)
            if(data["solution"]["solution_found"] == 0):
                elite_costs = data["solution"]["elite_cost_history"]
                wall_time = data["solution"]["wall_time_sec"]
                cost_histories.append(elite_costs)
                wall_histories.append(wall_time)

    cost_histories = np.array(cost_histories)
    wall_histories = np.array(wall_histories)
    return cost_histories, wall_histories

chimney_low_cost, chimney_wall_low = load(main_path + "chimney/climb_low/")
chimney_high_cost, chimney_wall_high = load(main_path + "chimney/climb_high/")
handrails_cost, handrails_wall = load(main_path + "handrails/")

### PLOT COSTS ###
chimney_low_cost = -chimney_low_cost
chimney_high_cost = -chimney_high_cost
handrails_cost = -handrails_cost

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

costs = [chimney_low_cost, chimney_high_cost, handrails_cost]
titles = ["Chimney (Low)", "Chimney (High)", "Handrails"]

fig, axes = plt.subplots(1, 3, sharey=False)

for i, (cost, ax) in enumerate(zip(costs, axes)):
    xx = np.arange(0, cost.shape[-1], 1)
    ax.plot(xx, np.mean(cost, axis=0), label="cost", color=colors[i])

    # Optional: Percentile fill
    plow = np.percentile(cost, 25, axis=0)
    phigh = np.percentile(cost, 75, axis=0)
    ax.fill_between(xx, plow, phigh, alpha=0.25, linewidth=0, color=colors[i])

    ax.set_title(titles[i])
    ax.set_xlabel("Iterations")
    if i == 0:
        ax.set_ylabel("Objective Value")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1)
    ax.set_axisbelow(True)

fig.suptitle("Time Minimization", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
plt.show()

fig.savefig("wall_costs.pdf")


# costs = [chimney_low_cost, chimney_high_cost, handrails_cost]
#
# fig = figure()
# ax = fig.add_subplot(111)
#
# # ax.set_yscale('log')
#
# for cost in costs:
#     xx = np.arange(0, cost.shape[-1], 1)
#     ax.plot(xx, np.mean(cost, axis=0), label="cost")
#     low = 25
#     up = 75
#     # plow = np.percentile(cost, low, axis=0)
#     # phigh = np.percentile(cost, up, axis=0)
#     # ax.fill_between(xx, plow, phigh, alpha=0.25, linewidth=0)#, color='#B22400')
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
# # ax.set_ylim([-5, 0])
#
# ax.legend(loc=8)
#
# fig.tight_layout()
#
# plt.show()
# exit()
# savefig("wall_costs" + ".pdf")
# # close()
