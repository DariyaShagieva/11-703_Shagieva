import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

n = 16
cla = np.int(np.sqrt(n))
x = []
y = []
clust = []


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def generateData(numberOfClassEl, numberOfClasses):
    data = []
    for classNum in range(numberOfClasses):
        centerX, centerY = random.random() * 10.0, random.random() * 10.0
        for rowNum in range(numberOfClassEl):
            data.append([[random.gauss(centerX, 1), random.gauss(centerY, 1)], classNum])
    for i in range(len(data)):
        x.append(data[i][0][0])
        y.append(data[i][0][1])
        clust.append(data[i][1])


generateData(n, cla)

x_min, x_max = min(x), max(x) - min(x)
y_min, y_max = min(y), max(y) - min(y)

x_new, y_new = (x_max - x_min) / 2, (y_max - y_min) / 2
new_clust = clust[0]

plt.scatter(x_new, y_new, color='b')
for i in range(0, len(clust)):
    clr = (clust[i] + 1) / cla
    plt.scatter(x[i], y[i], color=(clr, 0.2, clr ** 2))
plt.show()

distance = []

for i in range(len(x)):
    distance.append([[dist(x_new, y_new, x[i], y[i])], clust[i]])

distance.sort()

nghs = []

for i in range(cla):
    nghs.append(distance[i][1])

ng = Counter(nghs)

new_clust = max(set(nghs), key=lambda x: nghs.count(x))