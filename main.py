from desTree import DecisionTree
import numpy as np

X = np.array([[2.771244718, 1.784783929],
              [1.728571309, 1.169761413],
              [3.678319846, 2.812664299],
              [3.961043357, 2.61995032],
              [2.999208922, 2.209014212],
              [7.497545867, 3.162953546],
              [9.00220326, 3.339047188],
              [7.444242003, 0.476683375],
              [10.12493903, 3.234550982],
              [7, 3.2491514],
              [1.945158, 5.6948415],
              [6.2522525, 3.3255555519983761],
              [2.88858585, 1.25252525],
              [6.642287351, 3.319983761],
              [9.55555, 1]])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

model = DecisionTree(max_depth=3)
model.fit(X, y)
X2 = np.array([[2.5, 2.5], [1, 2], [3, 2]])

predictions = DecisionTree.predict(X2)
print("Предсказанные классы:", predictions)