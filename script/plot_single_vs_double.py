import numpy as np

eff_single_data = np.array([
    [0.909009, 0.910609, 0.910684, 0.910684, 0.910684, 0.910571],
    [0.922716, 0.922655, 0.922689, 0.92269, 0.92269, 0.922601],
    [1.14535, 1.1453, 1.14532, 1.14532, 1.14532, 1.14525],
    [1.19267, 1.19508, 1.19506, 1.19506, 1.19506, 1.19817],
])

iter_single_iter = [[24, 34, 45, 55, 64, 24], [12, 15, 19, 22, 25, 12],
                    [16, 19, 23, 26, 29, 16], [37, 48, 58, 71, 78, 37]]
eff_double_data = np.array([0.91057, 0.922601, 1.14525, 1.19816])

for row in range(6):
    for col in range(4):
        print("\\num{{{:.2e}}}".format(eff_single_data[col, row] -
                                       eff_double_data[col]),
              "&",
              "\\num{{{:d}}}".format(iter_single_iter[col][row]),
              "&",
              end="")
    print("\\\\")

single_time = np.array(
    [[
        7876, 11053, 14384, 17335, 20104, 4185, 5146, 6290, 7162, 8215, 5462,
        6467, 7474, 8628, 9364, 11912, 15533, 18467, 22490, 24736
    ],
     [
         7995, 10893, 14253, 17743, 20381, 4125, 5042, 6249, 7187, 8209, 5516,
         6290, 7769, 8724, 9497, 11834, 15236, 18220, 22409, 24618
     ],
     [
         8045, 10997, 14561, 17357, 20306, 4220, 5149, 6469, 7226, 8232, 5524,
         6299, 7549, 8546, 9437, 11954, 15362, 18552, 22421, 24606
     ],
     [
         8108, 10952, 14438, 17453, 20298, 4258, 5299, 6349, 7174, 8285, 5480,
         6365, 7625, 8476, 9417, 11892, 15283, 18442, 22386, 24535
     ],
     [
         7895, 11011, 14343, 17503, 20262, 4227, 5311, 6633, 7303, 8276, 5362,
         6559, 7681, 8768, 9432, 11897, 15341, 18442, 22412, 24598
     ],
     [
         8071, 11003, 14376, 17563, 20159, 4463, 5159, 6384, 7363, 8108, 5354,
         6360, 7571, 8526, 9496, 11960, 15430, 18531, 22432, 24503
     ],
     [
         7905, 11031, 14592, 17506, 20239, 4217, 5057, 6543, 7335, 5498, 5142,
         6409, 7763, 8454, 9498, 11959, 15414, 18446, 22454, 24453
     ],
     [
         8047, 11179, 14466, 17820, 20228, 4249, 5037, 6382, 7185, 8369, 5521,
         6389, 7621, 8781, 9527, 11962, 15318, 18565, 22468, 24612
     ],
     [
         7952, 10958, 14488, 17503, 20513, 4250, 5278, 6425, 7448, 8194, 5440,
         6401, 7620, 8437, 9249, 11979, 15345, 18426, 22324, 24853
     ],
     [
         7946, 10972, 14413, 17637, 20319, 4400, 5227, 6378, 7504, 8199, 5438,
         6411, 7555, 8422, 9719, 11961, 15321, 18541, 22462, 24703
     ]])
double_time = np.array([[12831, 7336, 9436, 19355], [12806, 7256, 9254, 19911],
                        [12980, 6981, 9548, 19434], [13201, 7332, 9268, 19035],
                        [13122, 6990, 9211, 19633], [12824, 7043, 9240, 19532],
                        [12854, 7026, 9230, 19450], [13294, 7595, 9353, 19148],
                        [13306, 7359, 9361, 19364], [13785, 7308, 9398,
                                                     19377]])
single_time_avg = (np.sum(single_time, axis=0) - np.max(single_time, axis=0) -
                   np.min(single_time, axis=0)) / (single_time.shape[0] - 2)
double_time_avg = (np.sum(double_time, axis=0) - np.max(double_time, axis=0) -
                   np.min(double_time, axis=0)) / (double_time.shape[0] - 2)

print(single_time_avg)
print(double_time_avg)

import plot_settings

fig = plot_settings.plt.figure(figsize=(plot_settings.A4_WIDTH,
                                        0.5 * plot_settings.A4_WIDTH))

cr_list = ["$0.01$", "$0.1$", "$10$", "$100$"]
x = np.arange(len(cr_list))

ax = fig.add_subplot()
width = 0.16
rects = ax.bar(x,
               double_time_avg,
               width=width,
               label="$1.0\mathrm{e}\!\!-\!\!5$(d)")
ax.bar_label(rects, fmt="")
current_single_time = single_time_avg[0::5]
rects = ax.bar(x + width,
               current_single_time,
               width=width,
               label="$1.0\mathrm{e}\!\!-\!\!5$(s)")
ax.bar_label(rects,
             labels=[
                 "x{:.1f}".format(ratio)
                 for ratio in current_single_time / double_time_avg
             ],
             fontsize=7)
current_single_time = single_time_avg[1::5]
rects = ax.bar(x + 2 * width,
               current_single_time,
               width=width,
               label="$1.0\mathrm{e}\!\!-\!\!6$(s)")
ax.bar_label(rects,
             labels=[
                 "x{:.1f}".format(ratio)
                 for ratio in current_single_time / double_time_avg
             ],
             fontsize=7)
current_single_time = single_time_avg[2::5]
rects = ax.bar(x + 3 * width,
               current_single_time,
               width=width,
               label="$1.0\mathrm{e}\!\!-\!\!7$(s)")
ax.bar_label(rects,
             labels=[
                 "x{:.1f}".format(ratio)
                 for ratio in current_single_time / double_time_avg
             ],
             fontsize=7)
current_single_time = single_time_avg[3::5]
rects = ax.bar(x + 4 * width,
               current_single_time,
               width=width,
               label="$1.0\mathrm{e}\!\!-\!\!8$(s)")
ax.bar_label(rects,
             labels=[
                 "x{:.1f}".format(ratio)
                 for ratio in current_single_time / double_time_avg
             ],
             fontsize=7)
current_single_time = single_time_avg[4::5]
rects = ax.bar(x + 5 * width,
               current_single_time,
               width=width,
               label="$1.0\mathrm{e}\!\!-\!\!9$(s)")
ax.bar_label(rects,
             labels=[
                 "x{:.1f}".format(ratio)
                 for ratio in current_single_time / double_time_avg
             ],
             fontsize=7)

ax.set_ylabel("Time (ms)")
ax.set_xlabel("$\kappa^\mathrm{inc}$")
ax.set_xticks(x + 2.5 * width, cr_list)

ax.legend(loc="upper center", ncol=3, fancybox=True, shadow=True)

plot_settings.plt.savefig("figs/single-vs-double.pdf", bbox_inches="tight")