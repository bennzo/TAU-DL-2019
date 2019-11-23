from facial_detect_loader import load
from facial_detect_model import train_model

if __name__ == '__main__':
    X, y, X_valid, y_valid = load()
    print(X.shape, X_valid.shape)
    # X_test, _, _, _ = load(True)
    train_model(X, y, X_valid, y_valid)
