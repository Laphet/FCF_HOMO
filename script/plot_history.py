def read_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            if (line.find("=====") != -1):
                new_chap = [1.0]
            relative_residual_begin = line.find("relative=")
            if (relative_residual_begin != -1):
                new_chap.append(float(line[relative_residual_begin + 9:-2]))
            rhs_begin = line.find("rhs=")
            rhs_end = line.find(",", rhs_begin)
            if (rhs_begin != -1 and rhs_end != -1):
                rhs = float(line[rhs_begin + 4:rhs_end])
            if (line.find("Reach") != -1):
                residual_begin = line.find("residual=")
                residual_end = line.find(" and")
                residual = float(line[residual_begin + 9:residual_end])
                new_chap.append(residual / rhs)
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
                                        plot_settings.A4_WIDTH),
                               layout="constrained")
gs = GridSpec(4, 4, figure=fig)

contrast_list = [0.01, 0.1, 10.0, 100.0]
dof_list = [400, 200, 100, 50]

for i in range(len(dof_list)):
    dof = dof_list[i]
    for j in range(len(contrast_list)):
        contrast = contrast_list[j]
        ax = fig.add_subplot(gs[i, j])
        data_idx = i * len(contrast_list) + j
        ax.plot(np.arange(len(fct_data[data_idx])),
                fct_data[data_idx],
                label="FCT")
        ax.plot(np.arange(len(ssor_05_data[data_idx])),
                ssor_05_data[data_idx],
                label="SSOR ($\omega=0.5$)")
        ax.plot(np.arange(len(ssor_10_data[data_idx])),
                ssor_10_data[data_idx],
                label="SSOR ($\omega=1.0$)")
        ax.plot(np.arange(len(ssor_15_data[data_idx])),
                ssor_15_data[data_idx],
                label="SSOR ($\omega=1.5$)")
        ax.plot(np.arange(len(icc_data[data_idx])),
                icc_data[data_idx],
                label="ICC")
        ax.set_yscale("log")
        if (not j == 0):
            ax.set_yticklabels([])
        ax.set_title("$\mathtt{dof}=" + str(dof) +
                     "^3$, $\kappa^\mathrm{inc}=" + str(contrast) + "$")
        handles, labels = ax.get_legend_handles_labels()

fig.legend(handles=handles,
           labels=labels,
           loc="lower center",
           bbox_to_anchor=(0.5, 1.02),
           ncol=5,
           fancybox=True,
           shadow=True)

plot_settings.plt.savefig("figs/convergence-history.pdf", bbox_inches="tight")
