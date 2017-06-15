import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def df_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class MLP:  # {
    def __init__(
            self,
            input_layer_neurons=2,
            hidden_layer_neurons=2,
            output_layer_neurons=1,
            f=sigmoid,
            df=df_sigmoid
            ):  # {
        # assign parameters to the object
        self.input_layer_neurons = input_layer_neurons
        self.hidden_layer_neurons = hidden_layer_neurons
        self.output_layer_neurons = output_layer_neurons
        self.f = f
        self.df = df

        # As stated in "Y. Bengio, X. Glorot, Understanding the difficulty
        # of training deep feedforward neuralnetworks, AISTATS 2010"
        # this is the optimal way to initialize the weights of a neuralnetwork
        # because it is the closest to a linear function in a tanh or sigmoid
        max = np.sqrt(6. / (input_layer_neurons + hidden_layer_neurons))
        # max = 0.5  # use to test later
        min = - max

        if (self.f == sigmoid):
            print("Using sigmoid")
            max *= 4
            min *= 4

        # h_neuron[neuron] => array of weights (W_h)
        self.hidden_layer_weights = (max-min) * np.random.rand(
                hidden_layer_neurons,
                input_layer_neurons
                ) + min
        # bias[neuron] => bias (b_h)
        self.hidden_layer_bias = (max-min) * np.random.rand(
                hidden_layer_neurons,
                1
                ) + min

        # o_neuron[n] => array of weights (W_o)
        self.output_layer_weights = (max-min) * np.random.rand(
                output_layer_neurons,
                hidden_layer_neurons
                ) + min

        # bias[neuron] => bias (b_o)
        self.output_layer_bias = (max-min) * np.random.rand(
                output_layer_neurons,
                1
                ) + min
    # ==end __init__ }

    # forward propagate the input through the network, producing the output
    def forward(self, X):  # {
        # from the input layer to hidden:
        f_h = np.zeros(self.hidden_layer_neurons)
        df_h = np.zeros(self.hidden_layer_neurons)
        # for each hidden neuron, calculate net, f(net) and df(net)/dnet
        for neuron in range(self.hidden_layer_neurons):
            W = self.hidden_layer_weights[neuron]
            b = self.hidden_layer_bias[neuron]
            # net = (input . W) + b
            net = X.dot(W) + b
            # if net < -1000:
            #     print('net for hid neuron [', neuron, ']: ', net)
            #     print('W is :', W)
            #     print('X is :', X)
            #     print('b is :', b)
            #     input()
            f_h[neuron] = self.f(net)
            df_h[neuron] = self.df(net)
        # ==end for

        # from the hidden layer to output:
        f_o = np.zeros(self.output_layer_neurons)
        df_o = np.zeros(self.output_layer_neurons)
        # for each output neuron, calculate net, f(net) and df(net)/dnet
        for neuron in range(self.output_layer_neurons):
            W = self.output_layer_weights[neuron]
            b = self.output_layer_bias[neuron]
            # net = (f_h . W) + b
            net = f_h.dot(W) + b
            # if net < -1000:
            #     print('net for out neuron [', neuron, ']: ', net)
            #     print('W is :', W)
            #     print('X is :', X)
            #     print('b is :', b)
            #     input()
            f_o[neuron] = self.f(net)
            df_o[neuron] = self.df(net)
        # ==end for
        return (f_h, df_h, f_o, df_o)
    # ==end forward }

    # backpropagate the error to update weights and learn
    def learn(self, X, expected_output, eta=0.1, threshold=1e-2):  # {
        squared_err = threshold * 2
        while squared_err >= threshold:
            squared_err = 0

            for test in range(len(X)):
                x = X[test]
                y = expected_output[test]

                (f_h, df_h, f_o, df_o) = self.forward(x)
                delta = y - f_o
                squared_err += np.sum(delta ** 2)

                # Apply generalized delta rule to the output layer
                # delta_o = length of output layer
                delta_o = np.multiply(delta, df_o)
                delta_h = np.multiply(
                        (delta_o @ self.output_layer_weights),
                        df_h
                        )
                # print('delta: ', delta)
                # print('delta_o: ', delta_o)
                # print('delta_h: ', delta_h)

                self.output_layer_weights = np.asarray(
                        self.output_layer_weights + (
                            eta * (np.asmatrix(delta_o).T @ np.asmatrix(f_h))
                            )
                        )
                self.output_layer_bias = np.asarray(self.output_layer_bias + (
                        eta * np.asmatrix(delta_o).T  # * [1,...]
                        ))
                # print('output_layer_weights: ', self.output_layer_weights)
                # print('output_layer_bias: ', self.output_layer_bias)

                # Apply generalized delta rule to the hidden layer
                # delta_h = length of hidden layer

                self.hidden_layer_weights = np.asarray(
                        self.hidden_layer_weights + (
                            eta * (np.asmatrix(delta_h).T @ np.asmatrix(x))
                            )
                        )
                self.hidden_layer_bias = np.asarray(self.hidden_layer_bias + (
                        eta * np.asmatrix(delta_h).T  # * [1,...]
                        ))
                # print('hidden_layer_weights: ', self.hidden_layer_weights)
                # print('hidden_layer_bias: ', self.hidden_layer_bias)
                # input()
            # ==end for test
            squared_err = squared_err / len(X)
            print('Avg Err: ', squared_err)
        # ==end while
    # ==end learn }
# ==end MLP }


def xor():
    X = np.array(
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
            )
    Y = np.array(
            [[0],
             [1],
             [1],
             [0]]
            )
    model = MLP(2, 4, 1)
    model.learn(X, Y, eta=0.1, threshold=1e-2)
    for x in X:
        (f_h, df_h, f_o, df_o) = model.forward(x)
        print(f_o)
    print(Y)


def get_y(y):
    r = np.zeros(10)
    r[y-1] = 1
    return r


def digit_recognizer():
    train_file = 'evens0123.csv'
    test_file = 'odds0123.csv'
    dataset = np.loadtxt(train_file, delimiter=',', skiprows=1)
    print('loaded train file!')
    X = (dataset[:, 1:len(dataset[0])] / 255)
    Y = dataset[:, 0]
    Y = np.array([get_y(int(y)) for y in Y])

    model = MLP(
            input_layer_neurons=X.shape[1],
            hidden_layer_neurons=10,
            output_layer_neurons=10
            )

    model.learn(X, Y, eta=0.1, threshold=1e-2)
    print('trained!')

    dataset = np.loadtxt(test_file, delimiter=',', skiprows=0)
    print('loaded test file!')
    X = (dataset[:, 1:len(dataset[0])] / 255)
    Y = dataset[:, 0]

    tries = len(X)
    success = 0
    for test in range(tries):
        x = X[test]
        y = int(Y[test])
        f_h, df_h, f_o, df_o = model.forward(x)
        if(np.round_(f_o[y-1]) == 1):
            success += 1
    print('got ', success, '/', tries, ': ', success*100/tries)


if __name__ == '__main__':
    digit_recognizer()
