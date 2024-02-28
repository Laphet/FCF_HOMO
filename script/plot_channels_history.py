def read_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            if line.find("=====") != -1:
                new_chap = [1.0]
            relative_residual_begin = line.find("relative=")
            if relative_residual_begin != -1:
                new_chap.append(float(line[relative_residual_begin + 9 : -2]))
            rhs_begin = line.find("rhs=")
            rhs_end = line.find(",", rhs_begin)
            if rhs_begin != -1 and rhs_end != -1:
                rhs = float(line[rhs_begin + 4 : rhs_end])
            if line.find("Reach") != -1:
                residual_begin = line.find("residual=")
                residual_end = line.find(" and")
                residual = float(line[residual_begin + 9 : residual_end])
                new_chap.append(residual / rhs)
            if line.find("homoCoeffZ") != -1:
                data.append(new_chap)
    return data


opt_data = read_file("reports/channels-type-0.log")
one_data = read_file("reports/channels-type-1.log")

import plot_settings
import numpy as np

fig = plot_settings.plt.figure(
    figsize=(0.8 * plot_settings.A4_WIDTH, 0.5 * plot_settings.A4_WIDTH),
    layout="constrained",
)
ax = fig.add_subplot()
ax.plot(np.arange(len(opt_data[0])), opt_data[0], label="opt($\Psi=1$)")
ax.plot(np.arange(len(one_data[0])), one_data[0], label="one($\Psi=1$)")
ax.plot(np.arange(len(opt_data[1])), opt_data[1], label="opt($\Psi=2$)")
ax.plot(np.arange(len(one_data[1])), one_data[1], label="one($\Psi=2$)")
ax.plot(np.arange(len(opt_data[2])), opt_data[2], label="opt($\Psi=3$)")
ax.plot(np.arange(len(one_data[2])), one_data[2], label="one($\Psi=3$)")
ax.set_yscale("log")
# ax.legend(fancybox=True, shadow=True)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(
#     handles=handles,
#     labels=labels,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.1),
#     ncol=3,
#     fancybox=True,
#     shadow=True,
# )
ax.legend(loc="upper center", ncol=3, fancybox=True, shadow=True)
plot_settings.plt.savefig("figs/channels-convergence-history.pdf", bbox_inches="tight")
