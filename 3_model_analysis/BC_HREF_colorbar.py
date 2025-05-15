import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define the color segments and corresponding values
colors = ["gray", "slateblue", "blue", "darkgreen", "green", "lightgreen", "yellow", "peru", "brown"]
bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Create a colormap and norm
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Create the colorbar
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cb = plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=ax, orientation='horizontal', ticks=bounds[:-1]
)

# Set tick labels
cb.set_ticks(bounds[:-1])
cb.ax.set_xticklabels([str(b) for b in bounds[:-1]])

plt.show()
plt.savefig('href_cb.png')
plt.close()