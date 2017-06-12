import sys
import numpy as np
import theano


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def df_sigmoid(x):
    return x * (1 - x)


class MLP: #{
    def __init__(self,
            input_layer_neurons=2,
            hidden_layer_neurons=2,
            output_layer_neurons=1,
            f=sigmoid,
            df=df_sigmoid
            ): #{
        # assign parameters to the object
        self.input_layer_neurons = input_layer_neurons;
        self.hidden_layer_neurons = hidden_layer_neurons;
        self.output_layer_neurons = output_layer_neurons;
        self.f = f;
        self.df = df;

        # As stated in "Y. Bengio, X. Glorot, Understanding the difficulty
        # of training deep feedforward neuralnetworks, AISTATS 2010"
        # this is the optimal way to initialize the weights of a neuralnetwork
        # because it is the closest to a linear function in a tanh or sigmoid
        max = np.sqrt(6. / (input_layer_neurons + hidden_layer_neurons))
        # max = 0.5 # use to test later
        min = - max

        if (False and self.f == sigmoid):
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
    #==end __init__ }

    # forward propagate the input through the network, producing the output
    def forward(self, input): #{
        # from the input layer to hidden:
        f_h = np.zeros(self.hidden_layer_neurons)
        df_h = np.zeros(self.hidden_layer_neurons)
        # for each hidden neuron, calculate net, f(net) and df(net)/dnet
        for neuron in range(self.hidden_layer_neurons):
            W = self.hidden_layer_weights[neuron]
            b = self.hidden_layer_bias[neuron]
            # net = (W . input) + b
            net = W.dot(input) + b
            f_h[neuron] = self.f(net)
            df_h[neuron] = self.df(net)
        #==end for

        # from the hidden layer to output:
        f_o = np.zeros(self.output_layer_neurons)
        df_o = np.zeros(self.output_layer_neurons)
        # for each output neuron, calculate net, f(net) and df(net)/dnet
        for neuron in range(self.output_layer_neurons):
            W = self.output_layer_weights[neuron]
            b = self.output_layer_bias[neuron]
            # net = (W . f_h) + b
            net = W.dot(f_h) + b
            f_o[neuron] = self.f(net)
            df_o[neuron] = self.df(net)
        #==end for
        return (f_h, df_h, f_o, df_o)
    #==end forward }
#==end MLP }


def main(argv):
    X = np.array(
            [[0,0],
             [0,1],
             [1,0],
             [1,1]]
            )
    Y = np.array(
            [[0],
             [1],
             [1],
             [0]]
            )
    model = MLP()
    for x in X:
        (f_h, df_h, f_o, df_o) = model.forward(x)
        print(f_o)
    print(Y)

if __name__ == '__main__':
    main(sys.argv)

