import numpy as np


def f(net):
    return 1 / (1 + np.exp(-net))


def df_dnet(net):
    return f(net) * (1 - f(net))


class my_little_poney:
    def __init__(
            self,
            input_layer_neurons=2,
            hidden_layer_neurons=2,
            output_layer_neurons=1,
            f_function=f,
            df_dnet_function=df_dnet
            ):
        # randomly initialize hidden layer weights
        self.hidden_weights = np.random.rand(
                hidden_layer_neurons,
                input_layer_neurons + 1
                ) - 0.5
        # randomly initialize output layer weights
        self.output_weights = np.random.rand(
                output_layer_neurons,
                hidden_layer_neurons + 1
                ) - 0.5

        self.input_layer_neurons = input_layer_neurons
        self.hidden_layer_neurons = hidden_layer_neurons
        self.output_layer_neurons = output_layer_neurons

        self.f = f_function
        self.df_dnet = df_dnet_function
    # end __init__

    def forward(self, input_values):

        input_values = [float(x) for x in np.transpose(input_values)]

        # initialize hidden neuron's fs and dfs
        # f_h = np.zeros(self.hidden_layer_neurons)
        # df_h = np.zeros(self.hidden_layer_neurons)
        f_h = [0]*self.hidden_layer_neurons
        df_h = [0]*self.hidden_layer_neurons

        # for each hidden neuron
        for neuron in range(self.hidden_layer_neurons):
            # net = f_h * hidden_weights + b
            net_h = np.dot(
                    # np.append(input_values, [1]),
                    input_values + [1],
                    self.hidden_weights[neuron, :]
                    )
            f_h[neuron] = self.f(net_h)
            df_h[neuron] = self.df_dnet(net_h)
        # end for

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
        # end for

        return (f_h, df_h, f_o, df_o)
    # end forward

    def backpropagation(self, X, Y, eta=0.1, threshold=1e-2):
        squared_err = 2 * threshold

        while squared_err > threshold:
            squared_err = 0
            for test in range(len(X)):
                x = X[test]  # get test line
                y = Y[test]  # get expected result line

                f_h, df_h, f_o, df_o = self.forward(x)

                f_h = [float(x) for x in f_h]

                delta = y - f_o  # difference between expected and real result

                # squared error += sum delta^2
                squared_err += np.sum(np.dot(delta, delta))

                # generalized delta rule
                delta_o = np.multiply(delta, df_o)

                w_length = self.hidden_layer_neurons - 1
                delta_h = np.multiply(
                        df_h,
                        np.dot(delta_o, self.output_weights[:, w_length])
                        )

                # update output layer weigths
                a = np.dot(
                        # np.transpose([np.append(f_h, [1])]), delta_o
                        np.transpose(np.matrix(f_h + [1])), delta_o
                        )

                self.output_weights += eta * np.transpose(a)
                # update hidden layer weigths

                x = [float(b) for b in np.transpose(x)]

                # c = np.transpose([np.append(x, [1])])

                self.hidden_weights += eta * np.transpose(
                        np.multiply(
                            delta_h,
                            np.transpose(np.matrix(np.transpose(x + [1])))
                            )
                        )

            # end for
            squared_err = squared_err/X.shape[0]
            print("Average squared error: " + str(squared_err))
        # end while
    # end backpropagation
# end my little poney class


def xor():
    dataset = np.loadtxt('xor.csv', skiprows=1)
    X = dataset[:, 0:len(dataset[0]) - 1]
    Y = dataset[:, len(dataset[0]) - 1:len(dataset[0])]

    print(X)
    print(Y)

    model = my_little_poney(2, 2, 1)
    model.backpropagation(np.matrix(X), np.matrix(Y), 0.5, 1e-2)

    for p in range(len(X)):
        x_p = X[p, :]
        y_p = Y[p, :]

        (f_h, df_h, f_o, df_o) = model.forward(x_p)
        print(x_p)
        print(y_p)
        print(f_o)


if __name__ == "__main__":
    xor()
