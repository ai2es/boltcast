import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib as mpl

# Define the color segments and corresponding values
colors = ["gray", "slateblue", "blue", "darkgreen", "green", "lightgreen", "yellow", "peru", "brown"]#HREF
bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Create a colormap and norm
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

cmap=plt.get_cmap("viridis")
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Create the horizontal colorbar
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cb = plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=ax, orientation='horizontal', ticks=bounds[:-1]
)

# Set tick labels
cb.set_ticks(bounds[:-1])
cb.ax.set_xticklabels([str(b) for b in bounds[:-1]],fontsize=24)

plt.show()
plt.savefig('viridis_cb.png')
plt.savefig('viridis_cb.pdf')
plt.close()


cmap = plt.get_cmap("viridis")
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Bin centers
labels = [str(b) for b in bounds[:]]
fig, ax = plt.subplots(figsize=(1, 6), layout='constrained')
cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax, 
    orientation='vertical')
cb.ax.set_yticklabels(labels, fontsize=24)

plt.savefig("viridis_cb_vert.png")
plt.savefig("viridis_cb_vert.pdf")
plt.show()
plt.close()