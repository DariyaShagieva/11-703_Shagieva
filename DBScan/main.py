import numpy as np
import matplotlib.pyplot as plt

eps = 6
minPts = 3
n = 200
x = np.random.randint(1, 100, n)
y = np.random.randint(1, 100, n)

flags = []


def dbscan():
    for i in range(0, n):
        pointCount = -1
        for j in range(0, n):
            if dist(x[i], y[i], x[j], y[j]) < eps:
                pointCount += 1
        if pointCount >= minPts:
            flags.append('g')
        else:
            flags.append('r')
    for i in range(0, n):
        if flags[i] != 'g':
            for j in range(0, n):
                if flags[j] == 'g':
                    if dist(x[i], y[i], x[j], y[j]) < eps:
                        flags[i] = 'y'
                        break

    cluster = [-1] * n
    alls = []
    for i in range(0, n):
        if flags[i] == 'g':
            alls.append(neighbors(i))

    for i in range(0, len(alls)):
        for j in range(0, len(alls)):
            if i != j and set(alls[i]).intersection(alls[j]) != set():
                alls[i].extend(alls[j])

    uniq = []
    for i in range(0, len(alls)):
        uniq.append(list(np.unique(alls[i])))
    realUniq = np.unique(uniq)

    for i in range(0, len(realUniq)):
        for j in range(0, len(realUniq[i])):
            cluster[realUniq[i][j]] = i

    x_cc = []
    y_cc = []
    for i in range(0, n):
        if flags[i] == 'r':
            x_cc.append(x[i])
            y_cc.append(y[i])
    k = len(np.unique(cluster)) - 1

    for i in range(1, len(flags)):
        if flags[i] == 'y':
            R = dist(x[i], y[i], x[0], y[0])
            for j in range(0, len(flags)):
                if flags[j] == 'g' and dist(x[i], y[i], x[j], y[j]) < eps and dist(x[i], y[i], x[j], y[j]) <= R:
                    cluster[i] = cluster[j]
    drow(cluster, k, x_cc, y_cc)


def neighbors(j):
    ngh = [j]
    for i in range(0, n):
        if j != i:
            if (flags[i] == 'g') and dist(x[j], y[j], x[i], y[i]) < eps:
                ngh.append(i)
    return ngh


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def drow(cluster, k, x_cc, y_cc):
    for i in range(0, len(cluster)):
        if cluster[i] >= 0:
            clr = (cluster[i]) / k
            plt.scatter(x[i], y[i], color=(clr, 0.2, clr ** 2))
    plt.scatter(x_cc, y_cc, color='red')
    plt.show()


if __name__ == '__main__':
    dbscan()
    for i in range(0, n):
        plt.scatter(x[i], y[i], color=flags[i])
    plt.show()
