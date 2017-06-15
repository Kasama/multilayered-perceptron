import os
import sys
import mlp
import pickle
import numpy as np
# import pandas as pd


def digit_recognizer_number_to_array(y):
    r = np.zeros(10)
    r[y-1] = 1
    return r


def digit_recognizer_array_to_number(arr):
    for i in range(len(arr)):
        if (arr[i] == 1):
            return (i+1) % 10
    return -1


def digit_recognizer(
        file, mode='train',
        hidden_neurons=10, eta=0.1, threshold=1e-2
        ):
    train_file = 'data/' + file + '/train.csv'
    test_file = 'data/' + file + '/test.csv'
    saved_model = 'trained/' + file + '-' + str(hidden_neurons) + '.mlp'

    if mode == 'test' and os.path.isfile(saved_model):
        model = pickle.load(open(saved_model, 'rb'))
    else:
        dataset = np.loadtxt(train_file, delimiter=',', skiprows=1)
        # dataset = pd.read_csv(
        #         train_file,
        #         sep=',',
        #         header=0,
        #         dtype=np.float64
        #         )
        print('loaded training file!')
        sys.stdout.flush()
        X = np.round(dataset[:, 1:len(dataset[0])] / 255)
        Y = dataset[:, 0]
        Y = np.array([digit_recognizer_number_to_array(int(y)) for y in Y])
        if mode == 'recover' and os.path.isfile(saved_model):
            model = pickle.load(open(saved_model, 'rb'))
        else:
            model = mlp.MLP(
                    input_layer_neurons=X.shape[1],
                    hidden_layer_neurons=hidden_neurons,
                    output_layer_neurons=10
                    )
            if not os.path.isdir('trained'):
                os.mkdir('trained')

        print('training network with ', hidden_neurons, ' hidden neurons:')
        sys.stdout.flush()
        model.learn(X, Y, eta, threshold, saved_model)
    print('trained!')
    sys.stdout.flush()

    dataset = np.loadtxt(test_file, delimiter=',', skiprows=0)
    print('loaded test file!')
    sys.stdout.flush()
    X = (dataset[:, 1:len(dataset[0])] / 255)
    Y = dataset[:, 0]

    tries = len(X)
    success = 0
    for test in range(tries):
        x = X[test]
        y = int(Y[test])
        f_h, df_h, f_o, df_o = model.feed_forward(x)
        if(digit_recognizer_array_to_number(np.round(f_o)) == y):
            success += 1
    print('got ', success, '/', tries, ': ', success*100/tries)
    sys.stdout.flush()


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        digit_recognizer(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        digit_recognizer(sys.argv[1], 'test')
