from facial_detect_loader import load
from facial_detect_model import train_model

import sys

def parse_args(argv):
    model_name = "simple"

    if len(argv) < 2:
        return model_name

    cmd = "-m <simple|conv>"

    if len(argv) >= 2:
        i = 1
        while i < len(argv):
            if argv[i] == "-m":
                assert (i + 1) < len(argv), "Model name is missing. Correct cmd: " + cmd 
                model_name = argv[i + 1]
                assert model_name in ['simple', 'conv'], \
                    "Bad model name: " + argv[i + 1] + " Use simple|conv"
                i += 1
            else:
                assert False, "bad command line. Correct cmd: " + cmd
            i += 1

    return model_name
    
if __name__ == '__main__':
    model_name = parse_args(sys.argv)
    X, y, X_valid, y_valid = load()
    # X_test, _, _, _ = load(True)
    train_model(X, y, X_valid, y_valid, model_name)
