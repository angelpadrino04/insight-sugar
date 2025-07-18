import numpy as np
import json
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)


class MLP:
    def __init__(self, input_size, hidden_layers=[16, 8], output_size=1,
                 hidden_activation='relu', output_activation='sigmoid',
                 learning_rate=0.01, random_seed=None):

        if random_seed:
            np.random.seed(random_seed)

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes)-1):
            if hidden_activation == 'relu':
                std = np.sqrt(2. / layer_sizes[i])
            else:
                std = np.sqrt(1. / layer_sizes[i])

            self.weights.append(np.random.randn(
                layer_sizes[i], layer_sizes[i+1]) * std)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)-1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)

            if self.hidden_activation == 'sigmoid':
                a = self._sigmoid(z)
            elif self.hidden_activation == 'tanh':
                a = self._tanh(z)
            else:  # ReLU por defecto
                a = self._relu(z)

            self.activations.append(a)

        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)

        if self.output_activation == 'sigmoid':
            a = self._sigmoid(z)
        elif self.output_activation == 'tanh':
            a = self._tanh(z)
        else:  # ReLU por defecto
            a = self._relu(z)

        self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y, output):
        m = X.shape[0]

        # Calcular gradiente de la capa de salida
        if self.output_activation == 'sigmoid':
            error = self._sigmoid_derivative(output) * (output - y)
        elif self.output_activation == 'tanh':
            error = self._tanh_derivative(output) * (output - y)
        else:  # ReLU
            error = self._relu_derivative(output) * (output - y)

        self.d_weights = [np.dot(self.activations[-2].T, error)]
        self.d_biases = [np.sum(error, axis=0, keepdims=True)]

        for i in range(len(self.weights)-1, 0, -1):
            if self.hidden_activation == 'sigmoid':
                error = np.dot(
                    error, self.weights[i].T) * self._sigmoid_derivative(self.activations[i])
            elif self.hidden_activation == 'tanh':
                error = np.dot(
                    error, self.weights[i].T) * self._tanh_derivative(self.activations[i])
            else:  # ReLU
                error = np.dot(
                    error, self.weights[i].T) * self._relu_derivative(self.activations[i])

            self.d_weights.insert(0, np.dot(self.activations[i-1].T, error))
            self.d_biases.insert(0, np.sum(error, axis=0, keepdims=True))

    def update_params(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def compute_loss(self, y, output):
        m = y.shape[0]
        loss = -np.mean(y * np.log(output + 1e-15) + (1 - y)
                        * np.log(1 - output + 1e-15))
        return loss

    def predict(self, X, threshold=0.5):
        output = self.forward(X)
        return (output >= threshold).astype(int)

    def evaluate(self, X, y, threshold=0.5):
        y_pred = self.predict(X, threshold)
        y_prob = self.forward(X)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        # Calcular ROC AUC
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }

    # Funciones de activación
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def save_weights(self, filename):
        weights_dict = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'config': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.output_size,
                'hidden_activation': self.hidden_activation,
                'output_activation': self.output_activation,
                'learning_rate': self.learning_rate
            }
        }
        with open(filename, 'w') as f:
            json.dump(weights_dict, f)

    @classmethod
    def load_weights(cls, filename):
        with open(filename, 'r') as f:
            weights_dict = json.load(f)

        config = weights_dict['config']
        model = cls(
            input_size=config['input_size'],
            hidden_layers=config['hidden_layers'],
            output_size=config['output_size'],
            hidden_activation=config['hidden_activation'],
            output_activation=config['output_activation'],
            learning_rate=config['learning_rate']
        )

        model.weights = [np.array(w) for w in weights_dict['weights']]
        model.biases = [np.array(b) for b in weights_dict['biases']]

        return model
