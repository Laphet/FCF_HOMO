import numpy as np

gpu_time_data = np.array([[
    21, 76, 603, 5093, 299, 79, 724, 5590, 30, 278, 1176, 9485, 38, 289, 2241,
    21631
],
                          [
                              21, 76, 570, 5298, 13, 81, 647, 5592, 31, 177,
                              1266, 9497, 53, 285, 2390, 21806
                          ],
                          [
                              21, 72, 577, 5368, 13, 82, 663, 5571, 31, 177,
                              1260, 9680, 38, 277, 2244, 21375
                          ],
                          [
                              24, 75, 580, 4890, 17, 270, 659, 5785, 31, 169,
                              1388, 9591, 37, 303, 2302, 21795
                          ],
                          [
                              22, 72, 567, 5559, 17, 316, 700, 6047, 33, 165,
                              1263, 9692, 54, 354, 2311, 21408
                          ],
                          [
                              21, 77, 568, 5307, 13, 88, 787, 5779, 31, 162,
                              1228, 9487, 51, 490, 2384, 21757
                          ],
                          [
                              21, 85, 676, 5299, 13, 82, 770, 5782, 24, 172,
                              1213, 9494, 38, 297, 2234, 21809
                          ],
                          [
                              21, 72, 558, 5174, 18, 313, 711, 5824, 32, 178,
                              1294, 9790, 39, 454, 2280, 21839
                          ],
                          [
                              21, 76, 580, 4894, 16, 299, 646, 6074, 33, 164,
                              1273, 9411, 37, 295, 2246, 21927
                          ],
                          [
                              21, 79, 558, 5564, 18, 83, 688, 5392, 23, 179,
                              1184, 9673, 49, 304, 2250, 21767
                          ]])

gpu_time_data_avg = (
    np.sum(gpu_time_data, axis=0) - np.max(gpu_time_data, axis=0) -
    np.min(gpu_time_data, axis=0)) / (gpu_time_data.shape[0] - 2)
# print(gpu_time_data_avg)

iter = np.array([19, 10, 9, 9, 15, 11, 11, 10, 29, 26, 22, 18, 48, 48, 43, 43])
dof_list = ["$64^3$", "$128^3$", "$256^3$", "$512^3$"]
cr_list = ["0.001", "0.01", "100", "1000"]

import plot_settings

fig, axs = plot_settings.plt.subplots(1,
                                      4,
                                      figsize=(plot_settings.A4_WIDTH,
                                               0.25 * plot_settings.A4_WIDTH),
                                      layout="constrained")
for i in range(4):
    x = np.arange(len(dof_list))  # the label locations
    # width = 0.25  # the width of the bars
    # rects = axs[i].bar(x,
    #                    gpu_time_data_avg[4 * i:4 * i + 4],
    #                    width,
    #                    label="Preparation",
    #                    color=plot_settings.NVIDIA_COLOR)
    # axs[i].bar_label(rects, fmt="")
    axs[i].plot(x,
                gpu_time_data_avg[4 * i:4 * i + 4],
                marker='.',
                color=plot_settings.NVIDIA_COLOR)
    axs[i].set_xticks(x, dof_list)
    axs[i].set_yscale("log")
    axs[i].set_title("$\kappa^\mathrm{inc}=" + cr_list[i] + "$")
    axs[i].set_xlabel("$\mathtt{dof}$")
    ax2 = axs[i].twinx()
    ax2.plot(x, iter[4 * i:4 * i + 4], marker='*', color="gray")
    ax2.set_ylim(0, 50)
    if i == 0:
        axs[i].set_ylabel("Time (ms)")
    if i == 3:
        ax2.set_ylabel("$\mathtt{iter}$")

plot_settings.plt.savefig("figs/gpu-cg-ball-cent.pdf", bbox_inches="tight")
