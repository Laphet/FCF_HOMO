import numpy as np

gpu_warmup_time = np.array([[701, 796, 671, 2426], [965, 890, 740, 1874],
                            [565, 663, 757, 1829], [817, 941, 675, 1768],
                            [560, 586, 1040, 1890], [651, 622, 1086, 2001],
                            [934, 848, 947, 1831], [887, 907, 704, 2568],
                            [815, 1018, 1054, 2256], [643, 695, 1093, 2289]])

gpu_solver_time = np.array([[106, 121, 104, 162], [111, 87, 109, 111],
                            [92, 114, 105, 117], [90, 103, 85, 106],
                            [98, 116, 95, 159], [85, 99, 147, 122],
                            [95, 87, 164, 183], [105, 94, 89, 134],
                            [117, 108, 94, 111], [95, 116, 141, 109]])

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

fig = plot_settings.plt.figure(figsize=(0.6 * plot_settings.A4_WIDTH,
                                        0.6 * plot_settings.A4_WIDTH),
                               layout="constrained")
ax = fig.add_subplot()
ax2 = ax.twinx()
x = np.arange(len(dof_list))  # the label locations
width = 0.25  # the width of the bars
rects = ax.bar(x, gpu_warmup_time_data, width, label="Preparation")
ax.bar_label(rects, fmt="")

ax.set_ylabel("Time (ms)")
ax.set_xlabel("$\mathtt{dof}$")
ax.set_xticks(x + width / 2, dof_list)
ax.legend(bbox_to_anchor=(0.0, 1.0),
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
                label="Preconditioner",
                color="#ff7f0e")
ax2.bar_label(rects, fmt="")

ax2.set_ylabel("Time (us)")
# ax.set_xlabel("$\mathtt{dof}$")
# ax.set_xticks(x, dof_list)
ax2.legend(bbox_to_anchor=(1.0, 1.0),
           loc="lower right",
           fancybox=True,
           shadow=True)
# ax.set_yscale("log")

plot_settings.plt.savefig("figs/precond-solver-gpu.pdf", bbox_inches="tight")
