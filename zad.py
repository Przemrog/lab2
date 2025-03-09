import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def perceptron(inputs, weights):
    activation = weights[0]
    for i in range(len(inputs) - 1):
        activation += weights[i + 1] * inputs[i]
    return 1 if activation >= 0 else 0


def train(data, lr, epoch):
    weights = [0.0 for i in range(len(data[0]))]
    error_per_epoch = []
    for ep in range(epoch):
        sum_error = 0.0
        for input in data:
            prediction = perceptron(input, weights)
            error = input[-1] - prediction
            sum_error += error ** 2
            weights[0] = weights[0] + lr * error
            for i in range(len(input) - 1):
                weights[i + 1] = weights[i + 1] + lr * error * input[i]
        error_per_epoch.append(sum_error)
        print('epoka=%d, wsp_uczenia=%.3f, err=%.3f' % (ep, lr, sum_error))

    plt.plot(range(epoch), error_per_epoch, marker='o')
    plt.title('zmiana bledu')
    plt.xlabel('epoka')
    plt.ylabel('suma bledow')
    plt.grid(True)
    plt.show()

    return weights

nieprocesowane_dane = [
    [25, 15, 3, "NIE", "NIE"],
    [35, 25, 5, "TAK", "TAK"],
    [45, 10, 2, "NIE", "NIE"],
    [28, 30, 7, "TAK", "TAK"],
    [32, 20, 4, "NIE", "NIE"],
    [40, 35, 6, "TAK", "TAK"],
    [50, 5, 1, "NIE", "NIE"],
    [30, 22, 5, "TAK", "TAK"],
    [27, 18, 3, "NIE", "NIE"],
    [38, 28, 8, "TAK", "TAK"],
    [29, 12, 2, "TAK", "NIE"],
    [42, 32, 6, "NIE", "TAK"],
    [33, 19, 4, "NIE", "NIE"],
    [48, 6, 1, "TAK", "NIE"],
    [31, 24, 5, "NIE", "TAK"],
    [26, 16, 3, "TAK", "NIE"],
    [37, 27, 7, "NIE", "TAK"],
    [44, 9, 2, "TAK", "NIE"],
    [36, 23, 5, "NIE", "TAK"],
    [24, 14, 3, "TAK", "NIE"]
]

data = np.array(nieprocesowane_dane)
X = data[:, :-1]
y = data[:, -1]

le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])
y = le.fit_transform(y)

X = X.astype(float)

scaler = StandardScaler()
X = scaler.fit_transform(X)

data = np.column_stack((X, y))

X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.3, random_state=42)

train_data = np.column_stack((X_train, y_train))

lr = 0.1
epoch = 20

weights = train(train_data, lr, epoch)

correct_predictions = 0
for inputs, true_value in zip(X_test, y_test):
    result = perceptron(np.append(inputs, 1), weights)
    if result == true_value:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test) * 100
print(f"\ndokladnosc na zbiorze testowym: {accuracy:.2f}%\n")

for inputs, true_value in zip(X_test, y_test):
    result = perceptron(np.append(inputs, 1), weights)  #
    print(f"oczekiwano={true_value}, wynik={result}")
