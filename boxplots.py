# Usage: python3 boxplots.py Chip12
# By adding Chip12, we will only plot data points with Chip12 in the video name
# If you run this with no arguments, then it will just plot everything

import numpy as np
from matplotlib import pyplot as plt
import sys

target_label = ""
if len(sys.argv) > 1:
	target_label = sys.argv[1]


px_diams = {}

with open("drop_diams_pixels.csv","r") as f:
	for line in f:
		vals = line.split(",")
		if target_label == "" or target_label in vals[0]:
			px_diams[vals[0]] = [float(x) for x in vals[1:]]

labels = sorted([x for x in px_diams])
reduced_labels = [x.split("_")[0].rsplit("/",1)[1] for x in labels]


bp = plt.boxplot([px_diams[x] for x in labels],labels=reduced_labels)

#for i in range(len(px_diams)):
#	y = px_diams[labels[i]]
#	x = np.random.normal(1+i, 0.04, size = len(y))
#	plt.plot(x,y, 'r.', alpha=0.2)

plt.show()

