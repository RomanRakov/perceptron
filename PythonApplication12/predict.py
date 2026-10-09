import argparse
import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def binary_crossentropy(y_true_onehot, y_pred_prob):
    eps = 1e-15
    y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred_prob), axis=1))

def predict():
    parser = argparse.ArgumentParser(description="Оценка обученной модели MLP")
    parser.add_argument('--weights', type=str, default='saved_model.npy', help="Путь к сохраненной модели")
    parser.add_argument('--data', type=str, default='dataset_split.npz', help="Путь к набору данных")
    args = parser.parse_args()

    try:
        data = np.load(args.data)
        X_test, y_test = data['X_test'], data['y_test']
    except FileNotFoundError:
        print(f"[ERROR] Файл данных '{args.data}' не найден. Сначала запустите split.py!")
        return
    except KeyError:
        print(f"[ERROR] В файле '{args.data}' не найден ключ 'X_test'. Перезапустите split.py!")
        return

    y_test_oh = np.eye(2)[y_test.squeeze()]

    try:
        model_data = np.load(args.weights, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"[ERROR] Файл модели '{args.weights}' не найден. Сначала запустите train.py!")
        return

    architecture = model_data['architecture']
    activations = model_data['activations']
    W_list = model_data['W']
    b_list = model_data['b']

    print(f"[INFO] Загружена модель с архитектурой: {architecture}")

    curr = X_test
    for i in range(len(W_list)):
        z = np.dot(curr, W_list[i]) + b_list[i]
        
        act_fn = activations[i]
        if act_fn == 'sigmoid':
            curr = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        elif act_fn == 'softmax':
            curr = softmax(z)

    y_pred_prob = curr

    loss = binary_crossentropy(y_test_oh, y_pred_prob)
    predictions = np.argmax(y_pred_prob, axis=1)
    accuracy = np.mean(predictions == y_test.squeeze())

    print("\n" + "="*40)
    print("      EVALUATION RESULTS (PREDICT)      ")
    print("="*40)
    print(f"Test Binary Cross-Entropy Loss : {loss:.4f}")
    print(f"Test Accuracy                  : {accuracy * 100:.2f}%")
    print("="*40)

if __name__ == '__main__':
    predict()