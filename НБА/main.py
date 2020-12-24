import pandas as pd

symptom = pd.read_csv('symptom.csv', delimiter=';')
disease = pd.read_csv('disease.csv', delimiter=';')

body = [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]

diseases = []


def nba():
    for i in range(len(disease) - 1):
        diseases.append(disease['количество пациентов'][i] / 303)

    p = 0
    result = ''
    for i in range(len(disease) - 1):
        chisl = diseases[i]
        znam = 1
        for j in range(1, len(body) + 1):
            if body[i] == 1:
                chisl *= symptom.iloc[i][1]
                znam *= 0.5
        p1 = chisl / znam
        if p < p1:
            p = p1
            result = disease.iloc[i][0]
            print(result)

    print('Итого: ', result)


if __name__ == '__main__':
    nba()
