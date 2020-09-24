import pandas as pd
import matplotlib.pyplot as plt


def read_data_from_csv(file):
    return pd.read_csv(file)


def count_survived(survived, Pclass):
    count_survive = {1: 0, 2: 0, 3: 0}
    for i in range(0, len(survived)):
        if survived[i] == 1:
            count_survive[Pclass[i]] += 1
    print(count_survive)
    return count_survive


def show_graph(data):
    plt.bar(data.keys(), data.values())
    plt.title("Количество выживших из разных классов")
    plt.xlabel("Класс")
    plt.ylabel("Количество выживших")
    plt.show()


if __name__ == '__main__':
    data = read_data_from_csv("titanic_data.csv")
    Pclass = list(data.Pclass)
    survived = list(data.Survived)
    count = count_survived(survived, Pclass)
    show_graph(count)
