from ann import ANN

from util import *

X, T = extract_data(data="mnist")

n = np.shape(X)[1]
model = ANN(data=X, target=T, split_percent=98, hidden_size=[int(n / 2)])

lr = np.float_power(10, -5)
lam = np.float_power(10, -0.5)
r = 1
epochs = 20
batch_size = 2 ** 8
activation = "relu"

model.train(gradient_type="descent", activation=activation,
            learning_rate={"type": "none",
                           "lr": lr},
            regularization={"R": r, "lambda": lam},
            batch_size=batch_size, epochs=epochs, plot_name="Regular")
model.train(gradient_type="descent", activation=activation,
            learning_rate={"type": "momentum",
                           "lr": lr,
                           "mu": 0.95},
            regularization={"R": r, "lambda": lam},
            batch_size=batch_size, epochs=epochs, plot_name="Momentum")
model.train(gradient_type="descent", activation=activation,
            learning_rate={"type": "nesterov",
                           "lr": lr,
                           "mu": 0.95},
            regularization={"R": r, "lambda": lam},
            batch_size=batch_size, epochs=epochs, plot_name="Nesterov")
model.train(gradient_type="descent", activation=activation,
            learning_rate={"type": "adagrad",
                           "lr": 1e-4,
                           "eps": 1e-8},
            regularization={"R": r, "lambda": 0.01},
            batch_size=batch_size, epochs=epochs, plot_name="Adagrad")
# model.train(gradient_type="descent", activation=activation,
#             learning_rate={"type": "rmsprop",
#                            "lr": 1e-6,
#                            "beta": 0.99,
#                            "eps": 1e-8},
#             regularization={"R": r, "lambda": lam},
#             batch_size=batch_size, epochs=epochs, plot_name="RMSProp")
model.train(gradient_type="descent", activation=activation,
            learning_rate={"type": "adam",
                           "lr": 1e-4,
                           "beta1": 0.9,
                           "beta2": 0.999,
                           "eps": 1e-8},
            regularization={"R": r, "lambda": lam},
            batch_size=batch_size, epochs=epochs, plot_name="Adam")

model.show_results()
