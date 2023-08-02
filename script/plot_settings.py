import matplotlib.pyplot as plt

from matplotlib import rc

rc("text", usetex=True)
rc("legend", fontsize=8)
A4_WIDTH = 8.27

# From https://www.schemecolor.com/monsoon-season.php
color_list = ["#B5C7CC", "#8BA1AD", "#587B89", "#244C66", "#407D6C", "#7DB290"]

NVIDIA_COLOR = "#76B900"

plt.style.use("seaborn-v0_8-paper")
