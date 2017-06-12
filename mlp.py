import sys
import numpy as np


def f(net):
    return 1 / (1 + np.exp(-net))


def df_dnet(net):
    return f(net) * (1 - f(net))


class mlp:
    def __init__(
            self,
            input_layer_neurons=2,
            hidden_layer_neurons=2,
            output_layer_neurons=1,
            f_function=f,
            df_dnet_function=df_dnet,
            hidden_weights=None,
            output_weights=None
            ):

        # randomly initialize hidden layer weights or use given
        if (hidden_weights is None):
            self.hidden_weights = mlp.generate_weights(
                    hidden_layer_neurons, input_layer_neurons + 1
                    )
        else:
            self.hidden_weights = hidden_weights

        # randomly initialize output layer weights
        if (output_weights is None):
            self.output_weights = mlp.generate_weights(
                    output_layer_neurons, hidden_layer_neurons + 1
                    )
        else:
            self.output_weights = output_weights

        # Assign layer sizes to instance variables
        self.input_layer_neurons = input_layer_neurons
        self.hidden_layer_neurons = hidden_layer_neurons
        self.output_layer_neurons = output_layer_neurons

        # Assign activation function and derivative to instance variables
        self.f = f_function
        self.df_dnet = df_dnet_function
    # ===== end __init__

    @staticmethod
    def generate_weights(x, y):
        return np.random.rand(x, y) - 0.5

    @staticmethod
    def train(
            X, Y,
            hidden_layer_neurons=2,
            f_function=f,
            df_dnet_function=df_dnet,
            eta=0.1,
            threshold=1e-2
            ):
        # normalize X and Y
        # X = np.matrix(X)
        # Y = np.matrix(Y)

        print("Xmatrix shape: ", X.shape)
        print("Ymatrix shape: ", Y.shape)

        # Guess input and output layer sizes
        input_layer_neurons = X.shape[1]
        output_layer_neurons = Y.shape[1]

        # Initialize MLP
        model = mlp(
                input_layer_neurons,
                hidden_layer_neurons,
                output_layer_neurons,
                f_function,
                df_dnet_function
                )

        # Train the network using given data
        model.backpropagation(X, Y, eta, threshold)

        return model
    # ===== end train

    @staticmethod
    def import_weights(hidden_file=None, output_file=None):
        if (hidden_file is None):
            hidden_file = 'hidden_layer_weights.npy'
        if (output_file is None):
            output_file = 'output_layer_weights.npy'
        hidden = np.load(hidden_file)
        output = np.load(output_file)

        hidden_neurons = hidden.shape[0]
        input_neurons = hidden.shape[1] - 1
        output_neurons = output.shape[0]

        if (hidden_neurons != (output.shape[1] - 1)):
            raise "sizes do not match"

        return mlp(
                input_layer_neurons=input_neurons,
                hidden_layer_neurons=hidden_neurons,
                output_layer_neurons=output_neurons,
                hidden_weights=hidden,
                output_weights=output
                )

    def export_weights(self, hidden_file=None, output_file=None):
        if (hidden_file is None):
            hidden_file = 'hidden_layer_weights.npy'
        if (output_file is None):
            output_file = 'output_layer_weights.npy'
        np.save(hidden_file, self.hidden_weights)
        np.save(output_file, self.output_weights)

    def predict(self, input_values):
        (f_h, df_h, f_o, df_o) = self.forward(input_values)
        return f_o

    def forward(self, input_values):

        # input_values = [float(x) for x in np.transpose(input_values)]

        # initialize hidden neuron's fs and dfs
        # f_h = np.zeros(self.hidden_layer_neurons)
        # df_h = np.zeros(self.hidden_layer_neurons)
        f_h = [0]*self.hidden_layer_neurons
        df_h = [0]*self.hidden_layer_neurons

        # for each hidden neuron
        for neuron in range(self.hidden_layer_neurons):
            # net = f_h * hidden_weights + b
            net_h = np.dot(
                    np.append(input_values, [1]),
                    # input_values + [1],
                    self.hidden_weights[neuron, :]
                    )
            f_h[neuron] = self.f(net_h)
            df_h[neuron] = self.df_dnet(net_h)
        # ===== end for

        # initialize output neuron's fs and dfs
        # f_o = np.zeros(self.output_layer_neurons)
        # df_o = np.zeros(self.output_layer_neurons)
        f_o = [0]*self.output_layer_neurons
        df_o = [0]*self.output_layer_neurons

        # for each output neuron
        for neuron in range(self.output_layer_neurons):
            # net = f_h * output_weights + b
            net_o = np.dot(
                    np.append(f_h, [1]),
                    self.output_weights[neuron, :]
                    )
            f_o[neuron] = self.f(net_o)
            df_o[neuron] = self.df_dnet(net_o)
        # ===== end for

        return (f_h, df_h, f_o, df_o)
    # ===== end forward

    def backpropagation(self, X, Y, eta=0.1, threshold=1e-2):
        squared_err = 2 * threshold

        while squared_err > threshold:
            squared_err = 0
            for test in range(len(X)):
                x = X[test]  # get test line
                y = Y[test]  # get expected result line

                f_h, df_h, f_o, df_o = self.forward(x)

                # f_h = [float(x) for x in f_h]

                delta = y - f_o  # difference between expected and real result

                # squared error += sum delta^2
                squared_err += np.sum(np.dot(delta, np.transpose(delta)))

                # generalized delta rule
                delta_o = np.multiply(delta, df_o)

                w_length = self.hidden_layer_neurons - 1
                delta_h = np.multiply(
                        df_h,
                        np.dot(delta_o, self.output_weights[:, w_length])
                        )

                # update output layer weigths
                a = np.dot(
                        np.transpose([np.append(f_h, [1])]), delta_o
                        # np.transpose(np.matrix(f_h + [1])), delta_o
                        )

                self.output_weights += eta * np.transpose(a)
                # update hidden layer weigths

                # x = [float(b) for b in np.transpose(x)]

                c = np.transpose([np.append(x, [[1]])])

                self.hidden_weights += eta * np.transpose(
                        np.multiply(
                            delta_h,
                            # np.transpose(np.matrix(np.transpose(x + [1])))
                            c
                            )
                        )

            # ===== end for
            squared_err = squared_err/X.shape[0]
            print("Average squared error: " + str(squared_err))
        # ===== end while
    # ===== end backpropagation
# ===== end my little poney class


def xor(train=True):
    dataset = np.loadtxt('xor.csv', skiprows=1, delimiter=',')
    X = dataset[:, 0:len(dataset[0]) - 1]
    Y = dataset[:, len(dataset[0]) - 1:len(dataset[0])]

    print(X)
    print(Y)

    if (train):
        model = mlp.train(X, Y, eta=0.2)
        print('exporting model')
        model.export_weights()
    else:
        model = mlp.import_weights()

    for p in range(len(X)):
        x = X[p, :]
        y = Y[p, :]

        o = model.predict(x)
        print(x)
        print(y)
        print(o)


def as_bit_array(number):
    return [
            (number/8) % 2,
            (number/4) % 2,
            (number/2) % 2,
            (number/1) % 2
            ]


def as_number(bit_array):
    num = bit_array[3]
    num += bit_array[2] * 2
    num += bit_array[1] * 4
    num += bit_array[0] * 8
    return num


def digit_recognizer_train(test='0123'):
    dataset = np.loadtxt('evens' + test + '.csv', delimiter=',')
    X = (np.round(dataset[:, 1:len(dataset[0])] / 255)) * 2 - 1
    Y = np.matrix([as_bit_array(x) for x in dataset[:, 0]])

    # Y = Y.reshape(len(Y), 1)

    print("X shape: ", X.shape)
    print("Y shape: ", Y.shape)

    # model = mlp.import_weights('bicalhohidden.npy', 'bicalhooutput.npy')
    # model.backpropagation(X, Y, eta=0.05, threshold=1e-2)

    model = mlp.train(X, Y, hidden_layer_neurons=50, eta=0.05, threshold=1e-2)
    print('Network trained, exporting weights')
    model.export_weights(
            'digit' + test + '_hidden.npy',
            'digit' + test + '_output.npy'
            )


def digit_recognizer_test(test='0123'):
    dataset = np.loadtxt('odds' + test + '.csv', delimiter=',')
    X = np.round(dataset[:, 1:len(dataset[0])] / 255)
    Y = dataset[:, 0]

    Y = Y.reshape(len(Y), 1)

    model = mlp.import_weights(
            'digit' + test + '_hidden.npy',
            'digit' + test + '_output.npy'
            )

    tries = len(X)
    success = 0
    for test in range(tries):
        out = model.predict(X[test])
        if (Y[test] == as_number([round(x) for x in out])):
            success += 1
    print('got ', success, '/', tries, ' right: ', success*100/tries, '%')


if __name__ == "__main__":
    if (sys.argv[1] == 'xor'):
        if (len(sys.argv) > 2 and sys.argv[2] == 'train'):
            xor(True)
        else:
            xor(False)
    elif (sys.argv[1] == 'digit'):
        if (sys.argv[2] == 'train'):
            digit_recognizer_train('0123')
        elif (sys.argv[2] == 'test'):
            digit_recognizer_test('0123')
