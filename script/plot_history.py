def read_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            if (line.find("=====") != -1):
                new_chap = [1.0]
            relative_residual_begin = line.find("relative=")
            if (relative_residual_begin != -1):
                new_chap.append(float(line[relative_residual_begin + 9:-2]))
            if (line.find("homoCoeffZ") != -1):
                data.append(new_chap)
    return data


fct_data = read_file("reports/convergence-history-fct.log")
ssor_10_data = read_file("reports/convergence-history-ssor-1-0.log")
ssor_05_data = read_file("reports/convergence-history-ssor-0-5.log")
ssor_15_data = read_file("reports/convergence-history-ssor-1-5.log")
icc_data = read_file("reports/convergence-history-icc.log")

import plot_settings
import numpy as np
from matplotlib.gridspec import GridSpec

fig = plot_settings.plt.figure(figsize=(plot_settings.A4_WIDTH,
                                        1.2 * plot_settings.A4_WIDTH),
                               layout="constrained")
gs = GridSpec(5, 4, figure=fig, height_ratios=[1.0, 5.0, 5.0, 5.0, 5.0])

contrast_list = [0.01, 0.1, 10.0, 100.0]
dof_list = [400, 200, 100, 50]

for i in range(len(dof_list)):
    dof = dof_list[i]
    for j in range(len(contrast_list)):
        contrast = contrast_list[j]
        ax = fig.add_subplot(gs[i + 1, j])
        data_idx = i * len(contrast_list) + j
        ax.plot(np.arange(len(fct_data[data_idx])), fct_data[data_idx])
        ax.plot(np.arange(len(ssor_10_data[data_idx])), ssor_10_data[data_idx])
        ax.plot(np.arange(len(ssor_05_data[data_idx])), ssor_05_data[data_idx])
        ax.plot(np.arange(len(ssor_15_data[data_idx])), ssor_15_data[data_idx])
        ax.plot(np.arange(len(icc_data[data_idx])), icc_data[data_idx])
        ax.set_yscale("log")
        ax.set_title()

plot_settings.plt.show()
