import numpy as np
from desTreeKNN import DecisionTreeKNN
from desTree import DecisionTree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from metrics import accuracy, f1_score, confusion_matrix


def plot_classification_results(X, y_true, y_pred, title="Классификация"):

    plt.figure(figsize=(10, 6))

    # Уникальные классы
    classes = np.unique(y_true)
    colors_true = ['blue', 'green', 'yellow', 'lime', 'navy', 'tan', 'aqua']  # Цвета для истинных классов
    colors_pred = ['red', 'pink', 'violet', 'firebrick', 'salmon', 'coral']  # Цвета для предсказанных классов

    # Визуализация истинных классов
    for i, cls in enumerate(classes):
        plt.scatter(X[y_true == cls, 0], X[y_true == cls, 1],
                    color=colors_true[i], label=f'Истинный класс {cls}', alpha=0.5, marker='o')

    # Визуализация предсказанных классов (только ошибочные предсказания)
    for i, cls in enumerate(classes):
        mask = (y_pred == cls) & (y_true != cls)  # Только ошибки
        if np.any(mask):
            plt.scatter(X[mask, 0], X[mask, 1],
                        color=colors_pred[i], label=f'Ошибочный класс {cls}', alpha=1.0, marker='x')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Генерация данных
    X, y = make_classification(n_samples=1000,
                               n_features=15,
                               n_informative=8,
                               n_redundant=0,
                               n_clusters_per_class=1,
                               n_classes=6,
                               random_state=42)

    # Разделение данных
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)


    model = DecisionTreeKNN(max_depth=3, crit='entropy')
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    model1 = DecisionTree(max_depth=3, crit='entropy')
    model1.fit(train_x, train_y)
    predictions1 = model1.predict(test_x)
    # Вывод результатов
    print("Истинные классы:\n", test_y)
    print("Предсказанные классы:\n", predictions)
    print("Точность KNN (accuracy):", accuracy(test_y, predictions))

    print("Истинные классы:\n", test_y)
    print("Предсказанные классы:\n", predictions1)
    print("Точность без KNN (accuracy):", accuracy(test_y, predictions1))
    nabl = int(1000*0.3)
    featu = 15
    plot_classification_results(test_x, test_y, predictions, title=f"Дерево с KNN, кол-во тестовых наблюдений = {nabl}, кол-во признаков = {featu}")
    plot_classification_results(test_x, test_y, predictions1, title=f"Дерево без KNN, кол-во тестовых наблюдений = {nabl}, кол-во признаков = {featu}")