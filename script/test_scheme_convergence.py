import plot_settings
import numpy as np

fig = plot_settings.plt.figure(figsize=(0.5 * plot_settings.A4_WIDTH,
                                        0.75 * plot_settings.A4_WIDTH))
ax = fig.add_subplot()

data_rtol9 = np.array(
    [[1.1188, 1.11737, 1.11688, 1.11651, 1.11661, 1.11652],
     [0.958059, 0.959911, 0.960645, 0.961019, 0.961107, 0.961194],
     [1.15966, 1.15734, 1.15602, 1.15518, 1.15513, 1.15492],
     [0.9052, 0.911941, 0.915234, 0.916845, 0.91745, 0.917842],
     [1.20798, 1.20343, 1.2002, 1.19838, 1.19793, 1.19745],
     [0.896265, 0.904186, 0.908154, 0.910086, 0.910837, 0.911309],
     [1.21558, 1.21053, 1.2069, 1.20488, 1.20434, 1.20382],
     [0.888529, 0.897563, 0.902179, 0.904419, 0.905312, 0.905861],
     [1.22199, 1.21649, 1.2125, 1.2103, 1.20968, 1.20912],
     [0.887521, 0.896706, 0.901411, 0.903693, 0.904606, 0.905165],
     [1.22281, 1.21725, 1.21322, 1.21099, 1.21036, 1.20979],
     [0.886707, 0.896016, 0.900793, 0.90311, 0.904039, 0.904606],
     [1.22356, 1.21794, 1.21386, 1.21161, 1.21098, 1.2104],
     [0.886523, 0.89586, 0.900654, 0.902978, 0.903911, 0.90448],
     [1.22363, 1.21801, 1.21393, 1.21167, 1.21104, 1.21046],
     [0.886505, 0.895844, 0.90064, 0.902965, 0.903898, 0.904467]])

n_list = [16, 32, 64, 128, 256, 512]
x_axis_label = list("${:d}^3$".format(n) for n in n_list)
x_axis = [4, 5, 6, 7, 8, 9]
cr_list = [
    5.0, 0.5, 10.0, 0.1, 50.0, 0.05, 100.0, 0.001, 500.0, 0.005, 1000.0, 0.001,
    10000.0, 0.0001, 100000.0, 0.00001
]
LEN = 6

for i in range(LEN):
    ax.plot(x_axis,
            data_rtol9[2 * i, :],
            label="$\kappa^\mathrm{inc}=" + "{:.1f}".format(cr_list[2 * i]) +
            "$",
            marker='.')
ax.legend(loc="upper center",
          bbox_to_anchor=(0.5, 1.08),
          ncol=3,
          fancybox=True,
          shadow=True)
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_label)
ax.set_xlabel("$\mathtt{dof}$")
ax.set_ylabel("$\kappa^\mathrm{eff}_z$")

plot_settings.plt.savefig("figs/test-scheme-convergence-1.pdf",
                          bbox_inches="tight")

ax.clear()
for i in range(LEN):
    ax.plot(x_axis,
            data_rtol9[2 * i + 1, :],
            label="$\kappa^\mathrm{inc}=" + str(cr_list[2 * i + 1]) + "$",
            marker='.')
ax.legend(loc="upper center",
          bbox_to_anchor=(0.5, 1.08),
          ncol=3,
          fancybox=True,
          shadow=True)
ax.set_xticks(x_axis)
ax.set_xticklabels(x_axis_label)
ax.set_xlabel("$\mathtt{dof}$")
ax.set_ylabel("$\kappa^\mathrm{eff}_z$")

plot_settings.plt.savefig("figs/test-scheme-convergence-2.pdf",
                          bbox_inches="tight")

ax.clear()
for i in range(LEN):
    ax.plot(x_axis[:-1],
            np.log2(np.abs(data_rtol9[2 * i, :-1] - data_rtol9[2 * i, -1])),
            label=r"$kappa=10^{:d}$".format(i + 1),
            marker='x')
ax.legend()
ax.grid()

ax.clear()
for i in range(LEN):
    ax.plot(x_axis[:-1],
            np.log2(
                np.abs(data_rtol9[2 * i + 1, :-1] -
                       data_rtol9[2 * i + 1, -1])),
            label=r"$kappa=10^{:d}$".format(-i - 1))
ax.legend()
ax.grid()
# plot_settings.plt.show()
