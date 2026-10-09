import argparse
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1.0 - a)

def binary_crossentropy(y_true_onehot, y_pred_prob):
    eps = 1e-15
    y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred_prob), axis=1))

class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='sigmoid'):
        self.activation_name = activation
        limit = np.sqrt(6.0 / input_dim)
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))

class MLP:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 2):
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], activation='sigmoid'))
        self.layers.append(DenseLayer(layer_sizes[-2], layer_sizes[-1], activation='softmax'))

    def forward(self, X):
        activations = [X]
        curr = X
        for layer in self.layers:
            z = np.dot(curr, layer.W) + layer.b
            if layer.activation_name == 'sigmoid':
                curr = sigmoid(z)
            elif layer.activation_name == 'softmax':
                curr = softmax(z)
            activations.append(curr)
        return activations

    def backward(self, activations, y_onehot, lr):
        m = y_onehot.shape[0]
        dz = (activations[-1] - y_onehot) / m
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_prev = activations[i]
            
            dW = np.dot(a_prev.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            
            if i > 0:
                da_prev = np.dot(dz, layer.W.T)
                dz = da_prev * sigmoid_derivative(a_prev)
            
            layer.W -= lr * dW
            layer.b -= lr * db

def train():
    parser = argparse.ArgumentParser(description="Обучение многослойного персептрона (MLP)")
    parser.add_argument('--layer', nargs='+', type=int, default=[24, 24, 24], help="Размеры скрытых слоев")
    parser.add_argument('--epochs', type=int, default=84, help="Количество эпох")
    parser.add_argument('--batch_size', type=int, default=8, help="Размер батча")
    parser.add_argument('--learning_rate', type=float, default=0.0314, help="Скорость обучения")
    parser.add_argument('--seed', type=int, default=42, help="Seed для фиксации случайности")
    args = parser.parse_args()

    np.random.seed(args.seed)

    try:
        data = np.load('dataset_split.npz')
        X_train, y_train = data['X_train'], data['y_train']
        X_valid, y_valid = data['X_valid'], data['y_valid']
    except FileNotFoundError:
        print("[ERROR] Файл 'dataset_split.npz' не найден. Сначала запустите split.py!")
        return

    y_train_oh = np.eye(2)[y_train.squeeze()]
    y_valid_oh = np.eye(2)[y_valid.squeeze()]

    architecture = [X_train.shape[1]] + args.layer + [2]
    model = MLP(architecture)

    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    n_samples = X_train.shape[0]

    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled_oh = y_train_oh[perm]

        for b in range(0, n_samples, args.batch_size):
            X_batch = X_shuffled[b:b+args.batch_size]
            y_batch = y_shuffled_oh[b:b+args.batch_size]
            acts = model.forward(X_batch)
            model.backward(acts, y_batch, args.learning_rate)

        train_acts = model.forward(X_train)
        valid_acts = model.forward(X_valid)

        t_loss = binary_crossentropy(y_train_oh, train_acts[-1])
        v_loss = binary_crossentropy(y_valid_oh, valid_acts[-1])

        t_acc = np.mean(np.argmax(train_acts[-1], axis=1) == y_train.squeeze())
        v_acc = np.mean(np.argmax(valid_acts[-1], axis=1) == y_valid.squeeze())

        train_losses.append(t_loss)
        valid_losses.append(v_loss)
        train_accs.append(t_acc)
        valid_accs.append(v_acc)

        print(f"epoch {epoch:02d}/{args.epochs:02d} " f"- loss: {t_loss:.4f} " f"- val_loss: {v_loss:.4f} " f"- acc: {t_acc:.4f} "f"- val_acc: {v_acc:.4f}")

    model_data = {
        'architecture': architecture,
        'activations': [l.activation_name for l in model.layers],
        'W': [l.W for l in model.layers],
        'b': [l.b for l in model.layers]
    }
    np.save('saved_model.npy', model_data)
    print("Saved model architecture and parameters to 'saved_model.npy'")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='training loss')
    ax1.plot(valid_losses, label='validation loss', linestyle='--')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_accs, label='training acc')
    ax2.plot(valid_accs, label='validation acc')
    ax2.set_title('Learning Curves (Accuracy)')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train()