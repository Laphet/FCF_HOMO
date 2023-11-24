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

cpu_time_data = np.array([[
    292, 537, 4253, 34618, 145, 575, 4710, 49922, 191, 1000, 7339, 54989, 371,
    1672, 12104, 114292
],
                          [
                              222, 537, 4158, 36636, 132, 553, 4652, 38287,
                              194, 1013, 7520, 55582, 322, 1607, 12681, 108517
                          ],
                          [
                              261, 540, 4503, 36598, 123, 587, 4770, 38463,
                              245, 990, 7500, 54880, 312, 1700, 12448, 102369
                          ],
                          [
                              251, 551, 4480, 34143, 122, 566, 4794, 38451,
                              186, 991, 7790, 64806, 293, 1638, 15179, 106132
                          ],
                          [
                              211, 551, 4215, 36766, 120, 555, 5331, 38212,
                              193, 1037, 7572, 52888, 288, 1607, 13745, 104716
                          ],
                          [
                              184, 543, 4523, 35958, 120, 563, 4776, 40737,
                              220, 1016, 7583, 61265, 360, 1737, 11099, 100693
                          ],
                          [
                              212, 546, 4265, 34929, 172, 644, 4761, 37351,
                              189, 1053, 7902, 80781, 329, 1619, 12032, 116424
                          ],
                          [
                              186, 563, 4210, 38628, 121, 575, 4494, 37531,
                              221, 1046, 7993, 65552, 386, 1645, 12501, 110391
                          ],
                          [
                              232, 569, 4144, 44723, 120, 559, 4567, 38368,
                              221, 1043, 7611, 53158, 311, 1600, 12795, 107735
                          ],
                          [
                              204, 546, 4325, 35874, 133, 551, 4833, 38700,
                              200, 976, 7653, 54707, 347, 1656, 12370, 110987
                          ]])

cpu_time_data_avg = (
    np.sum(cpu_time_data, axis=0) - np.max(cpu_time_data, axis=0) -
    np.min(cpu_time_data, axis=0)) / (cpu_time_data.shape[0] - 2)

iter = np.array([19, 10, 9, 9, 15, 11, 11, 10, 29, 26, 22, 18, 48, 48, 43, 43])
dof_list = ["$64^3$", "$128^3$", "$256^3$", "$512^3$"]
cr_list = ["0.001", "0.01", "100", "1000"]

import plot_settings

fig, axs = plot_settings.plt.subplots(2,
                                      2,
                                      figsize=(0.5*plot_settings.A4_WIDTH,
                                               0.5 * plot_settings.A4_WIDTH),
                                      layout="constrained")
for i in range(4):
    i_, j_ = divmod(i, 2)
    x = np.arange(len(dof_list))  # the label locations
    # width = 0.25  # the width of the bars
    # rects = axs[i].bar(x,
    #                    gpu_time_data_avg[4 * i:4 * i + 4],
    #                    width,
    #                    label="Preparation",
    #                    color=plot_settings.NVIDIA_COLOR)
    # axs[i].bar_label(rects, fmt="")
    axs[i_, j_].plot(x,
                cpu_time_data_avg[4 * i:4 * i + 4],
                marker='.',
                color=plot_settings.INTEL_COLOR,
                label="MKL-FFTW3")
    axs[i_, j_].plot(x,
                gpu_time_data_avg[4 * i:4 * i + 4],
                marker='.',
                color=plot_settings.NVIDIA_COLOR,
                label="CUDA")

    handles, labels = axs[i_, j_].get_legend_handles_labels()
    axs[i_, j_].set_xticks(x, dof_list)
    axs[i_, j_].set_yscale("log")
    axs[i_, j_].set_title("$\kappa^\mathrm{inc}=" + cr_list[i] + "$")
    axs[i_, j_].set_xlabel("$\mathtt{dof}$")
    ax2 = axs[i_, j_].twinx()
    ax2.plot(x, iter[4 * i:4 * i + 4], marker='*', color="gray")
    ax2.set_ylim(0, 50)
    if i in [0, 2]:
        axs[i_, j_].set_ylabel("Time (ms)")
    if i in [1, 3]:
        ax2.set_ylabel("$\mathtt{iter}$")
    if i in [0, 2]:
        ax2.set_yticklabels([])

fig.legend(handles=handles,
           labels=labels,
           loc="lower center",
           bbox_to_anchor=(0.5, 1.00),
           ncol=2,
           fancybox=True,
           shadow=True)

plot_settings.plt.savefig("figs/cg-ball-cent.pdf", bbox_inches="tight")
