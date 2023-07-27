import plot_settings
import numpy as np

fig = plot_settings.plt.figure(
    figsize=(0.5 * plot_settings.A4_WIDTH, 0.8 * plot_settings.A4_WIDTH)
)
ax = fig.add_subplot()

data_rtol9 = np.array(
    [
        [1.15602, 1.15518, 1.15513, 1.15492],
        [0.915234, 0.916845, 0.91745, 0.917842],
        [1.2069, 1.20488, 1.20434, 1.20382],
        [0.902179, 0.904419, 0.905312, 0.905861],
        [1.21322, 1.21099, 1.21036, 1.20979],
        [0.900793, 0.90311, 0.904039, 0.904606],
        [1.21386, 1.21161, 1.21098, 1.2104],
        [0.900654, 0.902978, 0.903911, 0.90448],
        [1.21393, 1.21167, 1.21104, 1.21046],
        [0.90064, 0.902965, 0.903898, 0.904467],
    ]
)

n_list = [64, 128, 256, 512]
x_axis = [6, 7, 8, 9]
LEN = 5
for i in range(LEN):
    ax.plot(
        x_axis,
        data_rtol9[2 * i, :] - data_rtol9[2 * i, -1],
        label="$10^{:d}$".format(i + 1),
        color=plot_settings.color_list[i],
    )
ax.legend()

# plot_settings.plt.savefig("test-scheme-convergence-1.pdf", bbox_inches="tight")
plot_settings.plt.show()
