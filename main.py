import numpy as np
from desTree import DecisionTree


if __name__ == "__main__":


    X = np.array([[2.5, 2.4],
                  [0.5, 0.7],
                  [3.3, 4.4],
                  [1.3, 1.1],
                  [3.0, 3.5],
                  [6.0, 5.0],
                  [5.5, 7.0]])
    y = np.array([0, 0, 1, 0, 1, 2, 2])  # Классы
    tree = DecisionTree(crit="entropy", max_depth=3, min_samples_split=2, knn_neighbor=3)
    tree.fit(X, y)
    x_test = ([[0.5, 1.0],
                  [5.2, 5.6],
                  [3.1, 4.0],
                  [7, 7]])

    predictions = tree.predict(x_test)

    for i, prediction in enumerate(predictions):
        print(f"Для X_test[{i}] = {x_test[i]}:  Prediction = {prediction}")


    # from sklearn.datasets import make_classification
    # from sklearn.model_selection import train_test_split
    #
    # # Генерация данных
    # X, y = make_classification(n_samples=100,  # количество образцов
    #                            n_features=5,  # количество признаков
    #                            n_informative=3,  # количество информативных признаков
    #                            n_redundant=0,  # количество избыточных признаков
    #                            n_clusters_per_class=1,
    #                            n_classes=3,  # количество классов
    #                            random_state=42)
    #
    # train_x, test_x, train_y, test_y = train_test_split(X, y)
    # model = DecisionTree(max_depth = 4, crit='gini')
    # model.fit(train_x, train_y)
    #
    # predictions = model.predict(test_x)
    # print(test_y)
    # print("Предсказанные классы:\n", predictions)
    # from metrics import accuracy, f1_score
    # print(accuracy(test_y, predictions))
    # print(f1_score(test_y, predictions))
    # predred = model.predict(train_x)
    # print(f1_score(train_y, predred))
