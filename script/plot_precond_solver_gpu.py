import numpy as np

# warmup_time = [[1540, 1472, 1604, 2907], [1487, 1554, 1598, 2552], [1429, 1533, 1853, 3019], [1383, 1481, 1864, 2997], [1458, 1524, 1721, 2593],
#                [1565, 1620, 1551, 2813], [1492, 1585, 1578, 2688], [1496, 1465, 1656, 2630], [1529, 1464, 1673, 2078], [1491, 1539, 1775, 2879]]

# solver_time = [[1, 8, 84, 917], [2, 6, 92, 920], [1, 7, 95, 911], [2, 6, 96, 913], [2, 8, 83, 666],
#                [2, 8, 83, 910], [2, 7, 97, 912], [1, 7, 87, 922], [2, 4, 97, 590], [2, 7, 94, 924]]

gpu_warmup_time = np.array([[1540, 1472, 1604, 2907], [1487, 1554, 1598, 2552], [1429, 1533, 1853, 3019], [1383, 1481, 1864, 2997], [1458, 1524, 1721, 2593],
               [1565, 1620, 1551, 2813], [1492, 1585, 1578, 2688], [1496, 1465, 1656, 2630], [1529, 1464, 1673, 2078], [1491, 1539, 1775, 2879]])

gpu_solver_time = np.array([[1, 8, 84, 917], [2, 6, 92, 920], [1, 7, 95, 911], [2, 6, 96, 913], [2, 8, 83, 666],
               [2, 8, 83, 910], [2, 7, 97, 912], [1, 7, 87, 922], [2, 4, 97, 590], [2, 7, 94, 924]])

gpu_warmup_time_data = (
    np.sum(gpu_warmup_time, axis=0) - np.max(gpu_warmup_time, axis=0) -
    np.min(gpu_warmup_time, axis=0)) / (gpu_warmup_time.shape[0] - 2)
gpu_solver_time_data = (
    np.sum(gpu_solver_time, axis=0) - np.max(gpu_solver_time, axis=0) -
    np.min(gpu_solver_time, axis=0)) / (gpu_solver_time.shape[0] - 2)
# print(gpu_warmup_time_data)
# print(gpu_solver_time_data)

dof_list = ["$64^3$", "$128^3$", "$256^3$", "$512^3$"]
import plot_settings

fig = plot_settings.plt.figure(figsize=(0.4 * plot_settings.A4_WIDTH,
                                        0.4 * plot_settings.A4_WIDTH),
                               layout="constrained")
ax = fig.add_subplot()
ax2 = ax.twinx()
x = np.arange(len(dof_list))  # the label locations
width = 0.25  # the width of the bars
rects = ax.bar(x, gpu_warmup_time_data, width, label="P.", color=plot_settings.color_list[0])
ax.bar_label(rects, fmt="")

ax.set_ylabel("Time (ms)")
ax.set_xlabel("$\mathtt{dof}$")
ax.set_xticks(x + width / 2, dof_list)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles=handles,
           labels=labels,
    bbox_to_anchor=(0.25, 1.0),
          loc="lower left",
          fancybox=True,
          shadow=True)
# ax.set_yscale("log")

# plot_settings.plt.savefig("figs/precond-solver-gpu-warmup.pdf", bbox_inches="tight")

# fig = plot_settings.plt.figure(figsize=(0.5 * plot_settings.A4_WIDTH,
#                                         0.5 * plot_settings.A4_WIDTH),
#                                layout="constrained")
# ax = fig.add_subplot()
# x = np.arange(len(dof_list))  # the label locations
rects = ax2.bar(x + width,
                gpu_solver_time_data,
                width,
                label="E.",
                color=plot_settings.color_list[-1])
ax2.bar_label(rects, fmt="")

ax2.set_ylabel("Time (ms)")
# ax.set_xlabel("$\mathtt{dof}$")
# ax.set_xticks(x, dof_list)
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles=handles,
           labels=labels,
    bbox_to_anchor=(0.75, 1.0),
           loc="lower right",
           fancybox=True,
           shadow=True)
# ax.set_yscale("log")

plot_settings.plt.savefig("figs/precond-solver-gpu.pdf", bbox_inches="tight")
