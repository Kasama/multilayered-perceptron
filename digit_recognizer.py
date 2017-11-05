import os
import sys
import mlp
import pickle
import numpy as np


class Number:  # {
    """{
    Create an representation of a number in both numeric and array form
    e.g: [0,0,0,1,0,0,0,0,0,0] = 4
         [0,0,0,0,0,0,0,0,0,1] = 0
         [1,0,0,0,0,0,0,0,0,0] = 1
    --- arguments ---
    what - 'number' if obj is a number and 'array' if obj is an array
    obj  - the object used to create the number representation
    }"""
    def __init__(self, what, obj):
        if (what == 'number'):
            self.number = obj
            self.array = np.zeros(10)
            self.array[self.number-1] = 1
        else:
            self.array = obj
            self.number = (self.array.argmax() + 1) % 10
        # ==end if
    # == end init

    @classmethod
    def from_number(klass, number):
        return klass('number', number)

    @classmethod
    def from_array(klass, arr):
        return klass('array', arr)
# == end Number class }


# train the model using train_file, eta and threshold.
# recover from saved_model and save results to saved_model {
def train_model(train_file, hidden_neurons, mode, eta, threshold, saved_model):
    dataset = np.loadtxt(train_file, delimiter=',', skiprows=1)
    print('loaded training file!')
    sys.stdout.flush()
    X = np.round(dataset[:, 1:len(dataset[0])] / 255)
    Y = dataset[:, 0]
    Y = np.array([Number.from_number(int(y)).array for y in Y])
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

    print('training network with', hidden_neurons, 'hidden neurons:')
    sys.stdout.flush()
    model.learn(X, Y, eta, threshold, saved_model)
    if (mode != 'predict'):
        print('trained!')
        sys.stdout.flush()
# == end train_model }


# test the model against test file and check answers to get score
def test_model(model, test_file):  # {
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
        if(Number.from_array(f_o).number == y):
            success += 1
    print('got ', success, '/', tries, ': ', success*100/tries)
    sys.stdout.flush()
# == end test_model }


# test the model against prediction file and output answers
def predict_model(model, predict_file):  # {
    dataset = np.loadtxt(predict_file, delimiter=',', skiprows=1)
    sys.stdout.flush()
    X = (dataset / 255)
    print('ImageId,Label')
    for test in range(len(X)):
        x = X[test]
        f_h, df_h, f_o, df_o = model.feed_forward(x)
        print(
                str(test + 1) +
                ',' +
                str(Number.from_array(f_o).number)
                )
        sys.stdout.flush()
# ==end predict_model }


""" {
Prepare the  dataset  and  run  MLP to  train or predict  results for the digit
recognizer MNIST test https://www.kaggle.com/c/digit-recognizer.
--- arguments ---
file - name of the dataset to use. Must be a valid directory under ./data
mode - must be one of 'train', 'test' or 'predict'.
     - 'train': force  training to start, even if there  is a partially trained
                file is present.
     - 'test' : if there is a already trained  saved model. Use it  against the
                test file and compare results. Otherwise  train the network and
                then test it.
     - 'predict' : if there is a  already trained saved  model, use  it against
                   the prediction file  to generate the output  to be submitted
                   to kaggle at https://www.kaggle.com/c/digit-recognizer.
hidden_neurons - the number of hidden layer neurons to use in the model.
eta - the step to be used in the backpropagation algorithm while training.
threshold - the maximum allowed error for the model while training.
} """


def digit_recognizer(
        file, mode='train',
        hidden_neurons=10, eta=0.1, threshold=1e-2
        ):  # {
    train_file = 'data/' + file + '/train.csv'
    test_file = 'data/' + file + '/test.csv'
    predict_file = 'data/' + file + '/predict.csv'
    saved_model = 'trained/' + file + '-' + str(hidden_neurons) + '.mlp'

    if (mode == 'test' or mode == 'predict') and os.path.isfile(saved_model):
        model = pickle.load(open(saved_model, 'rb'))
    else:
        model = train_model(
                train_file, hidden_neurons, mode,
                eta, threshold,
                saved_model
                )

    if (mode == 'test'):
        test_model(model, test_file)
    elif (mode == 'predict'):
        predict_model(model, predict_file)
# ==end digit_recognize }


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        digit_recognizer(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        digit_recognizer(sys.argv[1], 'test')
