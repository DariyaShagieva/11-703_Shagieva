import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans

n = 40


def setRandom(n):
    return np.random.randint(0, 100, n + 1)


def getPoints(n, x, y, z):
    points = []
    for i in range(n):
        points.append([x[i], y[i], z[i]])
    return points


def predict(features, weights):
    zz = np.dot(features, weights)
    return sigmoid(zz)


def sigmoid(zz):
    num = 1 / (1 + np.exp(-zz))
    return num


def update_weights(features, labels, weights, lr):
    N = len(features)

    predictions = predict(features, weights)
    gradient = np.dot(np.transpose(features), predictions - labels)
    gradient /= N
    gradient *= lr
    weights -= gradient
    return weights


def train(features, labels, weights, lr, iters):
    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)
    return weights


def decision(proc):
    return 'blue' if proc >= 0.5 else 'red'



def LR():
    x = setRandom(n)
    y = setRandom(n)
    z = setRandom(n)
    points = getPoints(n, x, y, z)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
    clusters = kmeans.labels_
    colors = ['red'] * n
    for i in range(n):
        if clusters[i] == 1:
            colors[i] = ('blue')

    xn = np.random.randint(0, 100)
    yn = np.random.randint(0, 100)
    zn = np.random.randint(0, 100)

    points.append([xn, yn, zn])
    x[len(x) - 1] = xn
    y[len(y) - 1] = yn
    z[len(z) - 1] = zn
    colors.append('green')
    weighss = train(points[:(len(points) - 1)], clusters, [1, 1, 1], 0.001, 10000)

    print("A: ", weighss[0], "  B: ", weighss[1], "  C: ", weighss[2], )

    print("point:", points[(len(points) - 1)])

    print("point:", decision(predict(points, weighss)[(len(points) - 1)]),
          "%=: ", '{:0.9f}'.format(predict(points, weighss)[(len(points) - 1)] * 100))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=colors)))

    tmp = np.linspace(0, 100, 50)
    xx, yx = np.meshgrid(tmp, tmp)

    zz = lambda x, y: ((weighss[0] * x - weighss[1] * y) / weighss[2])

    fig.add_trace(go.Surface(x=xx, y=yx, z=zz(xx, yx)))
    fig.show()


if __name__ == '__main__':
    LR()
