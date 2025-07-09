import numpy as np

np.random.seed(42)


class DeepNeuralNetwork:
    def __init__(self, layer_dims, alpha=0.01, epochs=10000):
        """
        layer_dims: list of layer sizes e.g. [2, 4, 3, 1] means:
            input layer: 2 neurons
            hidden layer 1: 4 neurons
            hidden layer 2: 3 neurons
            output layer: 1 neuron
        """
        self.layer_dims = layer_dims
        self.alpha = alpha
        self.epochs = epochs
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        parameters = {}
        L = len(self.layer_dims)
        for l in range(1, L):
            parameters[f"W{l}"] = np.random.randn(
                self.layer_dims[l - 1], self.layer_dims[l]
            ) * np.sqrt(2.0 / self.layer_dims[l - 1])
            parameters[f"b{l}"] = np.zeros((1, self.layer_dims[l]))
        return parameters

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        activations = {"A0": X}
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            Z = (
                activations[f"A{l - 1}"] @ self.parameters[f"W{l}"]
                + self.parameters[f"b{l}"]
            )
            A = self.sigmoid(Z)
            activations[f"Z{l}"] = Z
            activations[f"A{l}"] = A
        return activations

    def backward(self, X, y, activations):
        grads = {}
        L = len(self.layer_dims) - 1
        m = X.shape[0]
        A_final = activations[f"A{L}"]
        dA = -(y / A_final - (1 - y) / (1 - A_final))

        for l in reversed(range(1, L + 1)):
            A_prev = activations[f"A{l - 1}"]
            A = activations[f"A{l}"]
            dZ = dA * self.sigmoid_derivative(A)
            dW = A_prev.T @ dZ / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dA = dZ @ self.parameters[f"W{l}"].T

            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db

        for l in range(1, L + 1):
            self.parameters[f"W{l}"] -= self.alpha * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.alpha * grads[f"db{l}"]

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.mean(
            y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)
        )
        return loss

    def train(self, X, y):
        for i in range(self.epochs):
            activations = self.forward(X)
            y_pred = activations[f"A{len(self.layer_dims) - 1}"]
            loss = self.compute_loss(y_pred, y)
            self.backward(X, y, activations)
            if i % 1000 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        A_final = self.forward(X)[f"A{len(self.layer_dims) - 1}"]
        return (A_final > 0.5).astype(int)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


# (XOR Dataset)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define layer sizes: (input_layer-2 → hidden_layer1-6 → hidden_layer2-4 → output_layer-1)
layer_dims = [2, 6, 4, 1]
nn = DeepNeuralNetwork(layer_dims=layer_dims, alpha=0.1, epochs=10000)

nn.train(X, y)

print("\nFinal Predictions after training:\n", nn.predict(X))
print("Accuracy:", nn.accuracy(X, y))
