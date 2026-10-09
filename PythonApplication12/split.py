import argparse
import numpy as np
import pandas as pd

def split_data():
    parser = argparse.ArgumentParser(description="Разделение и стандартизация датасета")
    parser.add_argument('--csv', type=str, default='data.csv', help="Путь к исходному CSV")
    parser.add_argument('--seed', type=int, default=42, help="Seed для воспроизводимости")
    args = parser.parse_args()

    np.random.seed(args.seed)

    df = pd.read_csv(args.csv, header=None)
    
    y = (df.iloc[:, 1].values == 'M').astype(int)
    X = df.iloc[:, 2:].values.astype(float)

    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_end = int(0.70 * n_samples)
    valid_end = int(0.85 * n_samples)

    train_idx = indices[:train_end]
    valid_idx = indices[train_end:valid_end]
    test_idx = indices[valid_end:]

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_valid_raw, y_valid = X[valid_idx], y[valid_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std[std == 0] = 1e-8 

    X_train = (X_train_raw - mean) / std
    X_valid = (X_valid_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    np.savez('dataset_split.npz', 
             X_train=X_train, y_train=y_train,
             X_valid=X_valid, y_valid=y_valid,
             X_test=X_test, y_test=y_test,
             mean=mean, std=std)

    print("[SUCCESS] Данные успешно разделены и сохранены в 'dataset_split.npz':")
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Valid: {X_valid.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")

if __name__ == '__main__':
    split_data()