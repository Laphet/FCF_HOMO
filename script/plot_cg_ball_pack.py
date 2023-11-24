import numpy as np

iter = np.array([21, 12, 16, 33])
gpu_data = np.array([[11193, 6822, 8557, 16998], [11152, 7039, 8607, 17116],
                     [11145, 6843, 8785, 16766], [11109, 7032, 8741, 17095],
                     [11370, 6646, 8763, 16999], [10789, 6467, 9022, 16835],
                     [11175, 6992, 8565, 17045], [11431, 6700, 8818, 16830],
                     [11163, 6849, 8825, 16899], [11370, 6973, 8739, 16692]])
gpu_data_avg = (np.sum(gpu_data, axis=0) - np.max(gpu_data, axis=0) -
                np.min(gpu_data, axis=0)) / (gpu_data.shape[0] - 2)

cpu_data = np.array([[73880, 46057, 52797,
                      84340], [74798, 42605, 49766, 86149],
                     [82478, 43280, 53588,
                      84736], [56414, 39408, 49996, 89210],
                     [60634, 40208, 50089,
                      85348], [75674, 39993, 50892, 90350],
                     [72986, 42320, 49538,
                      85745], [69083, 43203, 51858, 82923],
                     [69699, 41721, 50962, 86524],
                     [62699, 43884, 49067, 84232]])
# print(gpu_data_avg)
# print(gpu_data_avg / iter)

cpu_data_avg = (np.sum(cpu_data, axis=0) - np.max(cpu_data, axis=0) -
                np.min(cpu_data, axis=0)) / (cpu_data.shape[0] - 2)
# print(cpu_data_avg)
# print(cpu_data_avg / iter)

cr_list = ["$0.01$", "$0.1$", "$10$", "$100$"]
import plot_settings

fig = plot_settings.plt.figure(figsize=(0.5 * plot_settings.A4_WIDTH,
                                        0.5 * plot_settings.A4_WIDTH),
                               layout="constrained")
ax = fig.add_subplot()
x = np.arange(len(cr_list))
width = 0.25
rects = ax.bar(x,
               gpu_data_avg,
               width,
               label="CUDA",
               color=plot_settings.NVIDIA_COLOR)
ax.bar_label(rects, fmt="")
rects = ax.bar(x + width,
               cpu_data_avg,
               width,
               label="MKL-FFTW3",
               color=plot_settings.INTEL_COLOR)
ax.bar_label(
    rects,
    labels=["x{:.1f}".format(ratio) for ratio in cpu_data_avg / gpu_data_avg],
    fontsize=7)
ax.set_ylabel("Time (ms)")
ax.set_xlabel("$\kappa^\mathrm{inc}$")
ax.set_xticks(x + width / 2, cr_list)
handles, labels = ax.get_legend_handles_labels()

ax2 = ax.twinx()
ax2.plot(x + width / 2, iter, marker='*', color="gray")
ax2.set_ylabel("$\mathtt{iter}$")

# ax.legend(ncol=2, fancybox=True, shadow=True)
fig.legend(handles=handles,
           labels=labels,
           loc="lower center",
           bbox_to_anchor=(0.5, 1.00),
           ncol=2,
           fancybox=True,
           shadow=True)

plot_settings.plt.savefig("figs/cg-ball-pack.pdf", bbox_inches="tight")
