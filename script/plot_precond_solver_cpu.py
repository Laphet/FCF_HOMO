import numpy as np

cpu_warmup_p1_time = np.array([[1224, 3536, 22622, 169761],
                               [1166, 3632, 21452, 155137],
                               [1113, 3481, 22607, 170493],
                               [1325, 3538, 22021, 165655],
                               [1458, 3295, 20523, 158373],
                               [1530, 3453, 22108, 171665],
                               [1447, 3421, 23233, 152386],
                               [873, 3271, 22283, 172331],
                               [1296, 3422, 21486, 164364],
                               [1411, 3281, 20650, 170082]])
cpu_warmup_p2_time = np.array([[289, 59, 268, 2000], [60, 60, 258, 1919],
                               [61, 60, 270, 1940], [62, 60, 277, 1955],
                               [60, 60, 268, 1937], [61, 60, 280, 1935],
                               [62, 60, 271, 1942], [63, 60, 263, 1985],
                               [67, 60, 264, 1912], [60, 58, 265, 1925]])
cpu_solver_p1_time = np.array([[22, 23, 212, 1363], [18, 22, 226, 1425],
                               [22, 26, 180, 1461], [18, 22, 176, 1624],
                               [16, 24, 211, 1468], [31, 23, 224, 1422],
                               [18, 25, 224, 1556], [12, 26, 219, 1356],
                               [19, 27, 216, 1493], [19, 24, 221, 1349]])
cpu_solver_p2_time = np.array([[81, 26, 291, 2535], [19, 26, 300, 2460],
                               [23, 26, 280, 2595], [26, 26, 288, 2561],
                               [20, 31, 258, 2527], [22, 26, 291, 2530],
                               [21, 29, 298, 2609], [17, 28, 304, 2355],
                               [26, 27, 301, 2423], [15, 25, 292, 2559]])

cpu_warmup_p1_data = (
    np.sum(cpu_warmup_p1_time, axis=0) - np.max(cpu_warmup_p1_time, axis=0) -
    np.min(cpu_warmup_p1_time, axis=0)) / (cpu_warmup_p1_time.shape[0] - 2)
cpu_warmup_p2_data = (
    np.sum(cpu_warmup_p2_time, axis=0) - np.max(cpu_warmup_p2_time, axis=0) -
    np.min(cpu_warmup_p2_time, axis=0)) / (cpu_warmup_p2_time.shape[0] - 2)
cpu_solver_p1_data = (
    np.sum(cpu_solver_p1_time, axis=0) - np.max(cpu_solver_p1_time, axis=0) -
    np.min(cpu_solver_p1_time, axis=0)) / (cpu_solver_p1_time.shape[0] - 2)
cpu_solver_p2_data = (
    np.sum(cpu_solver_p2_time, axis=0) - np.max(cpu_solver_p2_time, axis=0) -
    np.min(cpu_solver_p2_time, axis=0)) / (cpu_solver_p2_time.shape[0] - 2)
# print(cpu_warmup_p1_data)
# print(cpu_warmup_p2_data)
# print(cpu_solver_p1_data)
# print(cpu_solver_p2_data)

dof_list = ["$64^3$", "$128^3$", "$256^3$", "$512^3$"]
labels_list = ["Plan(P)", "Plan(E)"]

import plot_settings

fig = plot_settings.plt.figure(figsize=(0.5 * plot_settings.A4_WIDTH,
                                        0.5 * plot_settings.A4_WIDTH),
                               layout="constrained")
ax = fig.add_subplot()

x = np.arange(len(dof_list))  # the label locations
width = 0.25  # the width of the bars
rects = ax.bar(x, cpu_warmup_p1_data, width, label=labels_list[0])
ax.bar_label(rects, fmt="")
rects = ax.bar(x + width, cpu_warmup_p2_data, width, label=labels_list[1])
ax.bar_label(rects,
             labels=[
                 "x{:.1f}\%".format(ratio * 100)
                 for ratio in cpu_warmup_p2_data / cpu_warmup_p1_data
             ],
             fontsize=7)

ax.set_ylabel("Time (ms)")
ax.set_xlabel("$\mathtt{dof}$")
ax.set_xticks(x + width / 2, dof_list)
ax.legend(loc='upper left', ncols=2, fancybox=True, shadow=True)
ax.set_yscale("log")

plot_settings.plt.savefig("figs/precond-solver-cpu-warmup.pdf",
                          bbox_inches="tight")

fig = plot_settings.plt.figure(figsize=(0.5 * plot_settings.A4_WIDTH,
                                        0.5 * plot_settings.A4_WIDTH),
                               layout="constrained")
ax = fig.add_subplot()

x = np.arange(len(dof_list))  # the label locations
width = 0.25  # the width of the bars
rects = ax.bar(x, cpu_solver_p1_data, width, label=labels_list[0])
ax.bar_label(rects, fmt="")
rects = ax.bar(x + width, cpu_solver_p2_data, width, label=labels_list[1])
ax.bar_label(rects,
             labels=[
                 "x{:.1f}".format(ratio)
                 for ratio in cpu_solver_p2_data / cpu_solver_p1_data
             ],
             fontsize=7)

ax.set_ylabel("Time (ms)")
ax.set_xlabel("$\mathtt{dof}$")
ax.set_xticks(x + width / 2, dof_list)
ax.legend(loc='upper left', ncols=2, fancybox=True, shadow=True)
ax.set_yscale("log")

plot_settings.plt.savefig("figs/precond-solver-cpu-solver.pdf",
                          bbox_inches="tight")
