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
    "text.usetex": True,
    "figure.figsize": [1.6 * 4, 4],
}
rcParams.update(params)

main_path = "../results/"

def load(directory):
    f_list = glob.glob(directory + "*/*.json")
    all_hist = []
    success_hist = []
    for file in f_list:
        with open(file, 'r') as f:
            data = json.load(f)
            wall = data["solution"]["wall_time_sec"]
            all_hist.append(wall)
            if(data["solution"]["solution_found"] == 0):
                success_hist.append(wall)

    all_hist = np.array(all_hist)
    success_hist = np.array(success_hist)
    return all_hist, success_hist

chimney_low_all, chimney_low_success = load(main_path + "chimney/climb_low/")
chimney_high_all, chimney_high_success = load(main_path + "chimney/climb_high/")
handrails_all, handrails_success = load(main_path + "handrails/")

labels = ["Rails", "Ch. Low", "Ch. High"]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

bp = ax1.boxplot(
    [handrails_all.reshape((-1,)), chimney_low_all.reshape((-1,)), chimney_high_all.reshape((-1,))],
    notch=0,
    sym="b+",
    vert=1,
    whis=1.5,
    positions=None,
    widths=0.6,
)

for i in range(len(bp["boxes"])):
    box = bp["boxes"][i]
    box.set_linewidth(0)
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        boxPolygon = Polygon(boxCoords, facecolor=colors[i % len(colors)], linewidth=0)
        ax1.add_patch(boxPolygon)

for i in range(0, len(bp["boxes"])):
    bp["boxes"][i].set_color(colors[i])
    bp["whiskers"][i * 2].set_color(colors[i])
    bp["whiskers"][i * 2 + 1].set_color(colors[i])
    bp["whiskers"][i * 2].set_linewidth(2)
    bp["whiskers"][i * 2 + 1].set_linewidth(2)
    bp["fliers"][i].set( markerfacecolor=colors[i], marker="o", alpha=0.75, markersize=6, markeredgecolor="none")
    bp["medians"][i].set_color("black")
    bp["medians"][i].set_linewidth(3)
    for c in bp["caps"]:
        c.set_linewidth(0)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
ax1.tick_params(axis="x", direction="out")
ax1.tick_params(axis="y", length=0)
ax1.grid(axis="y", color="0.9", linestyle="-", linewidth=1)
ax1.set_axisbelow(True)
ax1.set_xticklabels(labels)
# ax.set_xlabel("X")
ax1.set_title("All Replicates")
ax1.set_ylabel("Wall-time (s)")

bp = ax2.boxplot(
    [handrails_success.reshape((-1,)), chimney_high_success.reshape((-1,)), chimney_low_success.reshape((-1,))],
    notch=0,
    sym="b+",
    vert=1,
    whis=1.5,
    positions=None,
    widths=0.6,
)

for i in range(len(bp["boxes"])):
    box = bp["boxes"][i]
    box.set_linewidth(0)
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        boxPolygon = Polygon(boxCoords, facecolor=colors[i % len(colors)], linewidth=0)
        ax2.add_patch(boxPolygon)

for i in range(0, len(bp["boxes"])):
    bp["boxes"][i].set_color(colors[i])
    bp["whiskers"][i * 2].set_color(colors[i])
    bp["whiskers"][i * 2 + 1].set_color(colors[i])
    bp["whiskers"][i * 2].set_linewidth(2)
    bp["whiskers"][i * 2 + 1].set_linewidth(2)
    bp["fliers"][i].set( markerfacecolor=colors[i], marker="o", alpha=0.75, markersize=6, markeredgecolor="none")
    bp["medians"][i].set_color("black")
    bp["medians"][i].set_linewidth(3)
    for c in bp["caps"]:
        c.set_linewidth(0)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
ax2.tick_params(axis="x", direction="out")
ax2.tick_params(axis="y", length=0)
ax2.grid(axis="y", color="0.9", linestyle="-", linewidth=1)
ax2.set_axisbelow(True)
ax2.set_xticklabels(labels)
ax2.set_title("Succesful Replicates")
# ax2.set_ylabel("Wall-time (s)")
# fig.subplots_adjust(left=-0.2)
# plt.show()
savefig("wall_times.pdf")

# ### PLOT COSTS ###
#
# talos_low = -talos_costs
# talos_high = -talos_costs
# print(talos_costs)
#
# fig = figure()
# ax = fig.add_subplot(111)
#
# ax.set_yscale('log')  # Set y-axis to log scale
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
# savefig("wall_costs" + ".pdf")
# # close()
