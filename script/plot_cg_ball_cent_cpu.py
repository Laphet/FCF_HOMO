import numpy as np

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

fig, axs = plot_settings.plt.subplots(1,
                                      4,
                                      figsize=(plot_settings.A4_WIDTH,
                                               0.25 * plot_settings.A4_WIDTH),
                                      layout="constrained")
for i in range(4):
    x = np.arange(len(dof_list))
    axs[i].plot(x,
                cpu_time_data_avg[4 * i:4 * i + 4],
                marker='.',
                color=plot_settings.INTEL_COLOR)
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

plot_settings.plt.savefig("figs/cpu-cg-ball-cent.pdf", bbox_inches="tight")