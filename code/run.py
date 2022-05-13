import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Для простоты работы
COLUMN = ["v" + str(i) for i in range(5000)]


def read_signals(filename):
    samples_count = 5000

    c = ['name', 'x', 'y']
    for i in range(0, samples_count):
        c.append(f'v{i}')
    c = c + ['cluster', 'p0', 'p1', 'p2', 'p3']

    df = pd.read_csv(filename, names=c, dtype=np.float32)
    df = df.set_index('name', drop=True)

    return df


def crete_cluster(df_file):
    condition = df_file[df_file["cluster"] >= 0]
    # подготовливая набор данных для обучения
    test_data_x = condition[COLUMN]
    test_data_y = condition["cluster"]

    # На основе алгоритма K-ближайших соседей
    # производим обучения на тестовых данных
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(test_data_x, test_data_y)
    return model


def predict_cluster_in_DataFrame(model, df_file):
    for i in range(len(df_file)):
        if df_file["cluster"][i] == -1:
            df_file["cluster"][i] = model.predict(df_file[i:i][COLUMN])
    return df


def write_signals(df, filename):
    df.to_csv(filename, header=False)


if __name__ == "__main__":
    # Example
    path = "../data/signals.csv"
    df = read_signals(path)
    print(df)
    model = crete_cluster(df)
    new_df = predict_cluster_in_DataFrame(model, df)
    print(new_df)
    write_signals(new_df, '../data/signals-out.csv')
