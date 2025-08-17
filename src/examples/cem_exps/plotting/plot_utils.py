import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 100
line_dark_colours = ["#033750", "#116069", "#885f1b", "#4b1e26", "#004b4f", "#59414a"]
line_colours = ["#056FA1", "#23C0D3", "#E0B266", "#963D4D", "#00969f", "#AA8B96"]
fill_colours = ["#58C6fA", "#8EE2EC", "#EFD8B2", "#D495A0", "#4FF5FF", "#D4C5CA"]
fill_light_colours = ["#08a8f4", "#55d4e3", "#e8c58c", "#be6071", "#00e9f7", "#bfa8b0"]

grid_colour = "#AFC0CA"
text_colour = "#221F21"


def create_fig():
    # Figure size in points
    fig_width_pt = 332
    fig_height_pt = 0.9 * 332
    # Convert to inches
    fig_width_in = fig_width_pt / 72
    fig_height_in = fig_height_pt / 72

    return plt.figure(figsize=(fig_width_in, fig_height_in))


def adjust_subplots(fig):
    fh, fw = fig.get_size_inches()
    margin_x = 6 / 72
    margin_bottom = 25 / 72
    margin_top = (10 + 15 + 11 + 17) / 72
    margin_w = 24 / 72
    margin_h = 15 / 72
    # Compute figure-relative coordinates
    l = margin_x / fw
    r = 1 - (margin_x / fw)
    b = margin_bottom / fh
    t = 1 - (margin_top / fh)
    ws = margin_w
    hs = margin_h

    fig.subplots_adjust(left=l, bottom=b, right=r, top=t, wspace=ws, hspace=hs)


def add_main_title(ax, text):
    x, y = pt_to_ax_coords(ax, 0, 10 + 10 + 15 + 15)
    fsize = 11
    weight = "bold"
    ax.text(
        x=x,
        y=y,
        s=text,
        transform=ax.transAxes,
        ha="left",
        fontsize=fsize,
        weight=weight,
        c=text_colour,
    )


def add_secondary_title(ax, text):
    x, y = pt_to_ax_coords(ax, 0, 10 + 10 + 15)
    fsize = 9.5
    weight = "normal"
    ax.text(
        x=x,
        y=y,
        s=text,
        transform=ax.transAxes,
        ha="left",
        fontsize=fsize,
        weight=weight,
        c=text_colour,
    )


def plot_line(ax, data, colour_idx, label=None, legend=False):
    x = np.arange(0, data.shape[0], 1)
    y = data
    ax.plot(x, y, c=line_colours[colour_idx], linewidth=3, label=label)
    if legend:
        add_legend(ax)


def plot_median_and_percentiles(
    ax, data, colour_idx, label=None, legend=False, low=25, high=75
):
    x = np.arange(0, data.shape[1], 1)
    y = np.median(data, axis=0)
    l = np.percentile(data, q=low, axis=0)
    h = np.percentile(data, q=high, axis=0)

    ax.plot(x, y, c=line_colours[colour_idx], linewidth=3, label=label)
    ax.fill_between(x, h, l, color=line_colours[colour_idx], alpha=0.25, linewidth=0)
    if legend:
        add_legend(ax)


def plot_boxplots(ax, data, labels):
    bp = ax.boxplot(
        data,
        notch=False,
        positions=[(i + 1) * 0.4 for i in range(0, len(data))],
        widths=0.3,
        labels=labels,
        patch_artist=True,
    )
    for i in range(0, len(bp["boxes"])):
        lc = line_dark_colours[i]
        llw = 3
        fc = line_colours[i]
        flw = 2
        bp["medians"][i].set(color=lc, linewidth=llw)
        bp["boxes"][i].set(color=fc)
        bp["whiskers"][i * 2].set(color=fc, linewidth=flw)
        bp["whiskers"][i * 2 + 1].set(color=fc, linewidth=flw)
        bp["fliers"][i].set(
            markerfacecolor=fc,
            marker="o",
            markersize=6,
            markeredgecolor="none",
        )
        for c in bp["caps"]:
            c.set_linewidth(0)

    ax.legend(
        [box for box in bp["boxes"]],
        labels,
        loc=(0, 1.03),
        ncol=len(ax.get_lines()),
        frameon=False,
        handleheight=0.7,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.9,
        labelcolor=text_colour,
        fontsize=9,
    )


def add_legend(ax):
    ax.legend(
        loc=(0, 1.03),
        ncol=len(ax.get_lines()),
        frameon=False,
        handleheight=0.7,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.9,
        labelcolor=text_colour,
        fontsize=9,
    )


def add_x_ticks(ax, xticks, xlabels=None):
    if xlabels is None:
        xlabels = xticks
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax.set_xticks(ticks=xticks, labels=xlabels, c=text_colour, fontsize=7)
    ax.margins(x=0.1)


def add_y_ticks(ax, yticks, ylabels=None):
    if ylabels is None:
        ylabels = yticks
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(which="major", axis="y", c=grid_colour, linewidth=0.5)

    ax.set_yticks(ticks=yticks, labels=ylabels, c=text_colour)
    ax.set_yticklabels(ylabels, ha="right", va="bottom", c=text_colour, fontsize=7)
    ax.tick_params(axis="y", direction="in", pad=-0.5, length=0)
    ax.yaxis.tick_right()


def pt_to_ax_coords(ax, pt_x, pt_y):
    inches_x = pt_x / 72
    inches_y = pt_y / 72
    x = inches_x / ax.figure.get_size_inches()[0]
    y = 1 + (inches_y / ax.figure.get_size_inches()[1])
    return x, y
