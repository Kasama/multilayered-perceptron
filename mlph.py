import threading
import numpy as np
import pickle
import sys
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def df_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class MLP:  # {
    def __init__(
            self,
            input_layer_neurons,
            hidden_layer_neurons,
            hidden_layers,
            output_layer_neurons,
            f=sigmoid,
            df=df_sigmoid
            ):  # {
        # assign parameters to the object
        self.input_layer_neurons = input_layer_neurons
        self.hidden_layer_neurons = hidden_layer_neurons
        self.hidden_layers = hidden_layers
        self.output_layer_neurons = output_layer_neurons
        self.f = f
        self.df = df
        self.squared_err = -1

        """
        As  stated in  "Y. Bengio, X. Glorot, Understanding  the  difficulty of
        training  deep feedforward  neuralnetworks, AISTATS 2010"  this is  the
        optimal  way to initialize the weights of a neuralnetwork because it is
        the closest to a linear function  in a `tanh` or `sigmoid`. If we use a
        `sigmoid`  function, the min/max  values should  be multiplied  by 4 to
        better accommodate the curve
        """
        max = np.sqrt(6. / (input_layer_neurons + hidden_layer_neurons))
        # max = 0.5  # use to test later
        min = - max

        if (self.f == sigmoid):
            sys.stdout.flush()
            max *= 4
            min *= 4

        # h_neuron[neuron] => array of weights (W_h)
        self.hidden_layer_weights = (max-min) * np.random.rand(
                hidden_layers,
                hidden_layer_neurons,
                input_layer_neurons) + min

        # bias[neuron] => bias (b_h)
        self.hidden_layer_bias = (max-min) * np.random.rand(
                hidden_layers,
                hidden_layer_neurons,
                1) + min

        # o_neuron[n] => array of weights (W_o)
        self.output_layer_weights = (max-min) * np.random.rand(
                output_layer_neurons,
                hidden_layer_neurons) + min

        # bias[neuron] => bias (b_o)
        self.output_layer_bias = (max-min) * np.random.rand(
                output_layer_neurons,
                1) + min
    """ feed forward the input to get the output{
    feed forward to propagate the input through the network,
    producing activation functions and their derivatives
    --- arguments ---
    X: one test case, must be the same size as input_neurons
    } """
    def feed_forward(self, X):  # {
        # from the input layer to hidden:
        f_h = np.zeros(self.hidden_layer_neurons)
        df_h = np.zeros(self.hidden_layer_neurons)
        # for each layer
        for layer in range(self.hidden_layers):
            # for each hidden neuron, calculate net, f(net) and df(net)/dnet
            for neuron in range(self.hidden_layer_neurons):
                W = self.hidden_layer_weights[layer][neuron]
                b = self.hidden_layer_bias[layer][neuron]
                # net = (input . W) + b
                net = X.dot(W) + b
                f_h[neuron] = self.f(net)
                df_h[neuron] = self.df(net)

            # from the hidden layer to the next:
            f_o = np.zeros(self.output_layer_neurons)
            df_o = np.zeros(self.output_layer_neurons)
            
            # from the hidden layer to output:
            f_o = np.zeros(self.output_layer_neurons)
            df_o = np.zeros(self.output_layer_neurons)
            # for each output neuron, calculate net, f(net) and df(net)/dnet
            for neuron in range(self.output_layer_neurons):
                W = self.output_layer_weights[neuron]
                b = self.output_layer_bias[neuron]
                # net = (f_h . W) + b
                net = f_h.dot(W) + b
                f_o[neuron] = self.f(net)
                df_o[neuron] = self.df(net)
        return (f_h, df_h, f_o, df_o)

    """ calculate deltas {
    Calculate the  delta for  each layer, based  on the  local error  using the
    generalized delta rule to solve the gradient descendent
    --- arguments ---
    expected - the expected output. Must  be of length  equal to the  number of
               output_layer_neurons
    out - the actual output  layer values. Must be the  same length as expected
    df_h - hidden layer  activation derivatives. Must be the same length as the
           number of hidden_layer_neurons
    df_o - output layer  activation derivatives. Must be the same length as the
           number of output_layer_neurons
    } """
    def get_deltas(self, expected, out, df_h, df_o):  # {
        # delta = expected_out - actual_out
        delta = expected - out

        # Apply generalized delta rule to the output layer
        delta_o = np.multiply(delta, df_o)
        # Apply generalized delta rule to the hidden layer
        delta_h = np.multiply(
                (delta_o @ self.output_layer_weights),
                df_h
                )
        return (delta, delta_o, delta_h)
    # ==end get_deltas }

    """ learn using the backpropagation algorithm {
    Run feed_forward for  the whole  dataset, updating the error every round to
    get an average  error. This total  will be used to determine how  good  the
    network is at solving  the problem. When  it is good (error < threshold) we
    can stop training.
    Every feed forward round, the weights of each  neuron get updated using the
    backpropagation  algorithm, witch uses  the gradient descendant to find the
    direction of the minimum  for the error function. In the  case of a average
    square error, this gradient is calculated by multiplying each error by  its
    correspondent activation function derivative
    --- arguments ---
    X - An array of test  cases. Each X[i] must  be a test case of length equal
        to the number of input_neurons
    expected_output - An array of  expected results, each  corresponding to one
                      test  case in X. One test  case must have length equal to
                      the number of output_neurons
    eta - Learning  rate. Is  a factor to  how much  the neuron's  weights  are
          going to be changed. Must be in range (0,1]
    threshold - Error threshold.  Indicates  how low  the error  must be  to be
                for the network to be considered 'good'
    } """
    def learn(self, X, expected_output, eta=0.1, threshold=1e-4, file='', verbose=True):  # {
        if self.squared_err == -1:
            self.squared_err = threshold * 2
        else:
            print('recovered with avg err: ', self.squared_err)
            sys.stdout.flush()
        # while we are not good enough
        while self.squared_err >= threshold:
            self.squared_err = 0
            for test in range(len(X)):
                x = X[test]
                y = expected_output[test]

                # run feed_forward for the test case to get activation
                # value and derivatives for all layers
                (f_h, df_h, f_o, df_o) = self.feed_forward(x)

                # calculate gradient descendent using generalized delta rule
                delta, delta_o, delta_h = self.get_deltas(y, f_o, df_h, df_o)

                # Cost function squared error = sum(||delta||^2)
                self.squared_err += np.sum(delta ** 2)

                # Update weights in the output layer
                # w'[i] = w[i] + (eta * (sum_j(delta_o[i][j] * f_h[i][j])))
                self.output_layer_weights = np.asarray(
                        self.output_layer_weights + (
                            eta * (np.asmatrix(delta_o).T @ np.asmatrix(f_h))
                            )
                        )
                # Update bias in the output layer
                # b'[i] = b[i] + (eta * delta_o[i])
                self.output_layer_bias = np.asarray(self.output_layer_bias + (
                        eta * np.asmatrix(delta_o).T  # * [1,...]
                        ))

                # Update weights in the hidden layer
                # w'[i] = w[i] + (eta * (sum_j(delta_h[i][j] * x[i][j])))
                for layer in range(self.hidden_layers):
                    self.hidden_layer_weights[layer] = np.asarray(
                            self.hidden_layer_weights[layer] + (
                                eta * (np.asmatrix(delta_h).T @ np.asmatrix(x))
                                )
                            )
                    # Update bias in the hidden layer
                    # b'[i] = b[i] + (eta * delta_h[i])
                    self.hidden_layer_bias[layer] = np.asarray(self.hidden_layer_bias[layer] + (
                            eta * np.asmatrix(delta_h).T  # * [1,...]
                            ))
            # ==end for test
            self.squared_err = self.squared_err / len(X)
            if file != '':
                pickle.dump(self, open(file, 'wb'))
            if verbose: print('Avg Err: ', self.squared_err)
            sys.stdout.flush()
        # ==end while
    # ==end learn }
# ==end MLP }

if __name__ == '__main__':
    bucket = {}
    tc = [0]

    def time_h_mlp(h):
        mlp = MLP(2, 3, h, 1)
        x = np.array([[1,0],[0,0],[0,1],[1,1]])
        y = np.array([1,0,1,0])
        s = time.time()
        mlp.learn(x,y,threshold=1e-2,verbose=False)
        if h not in bucket: bucket[h]=[]
        bucket[h].append(time.time() - s)
        tc[0]-=1
        print(h, np.average(bucket[h]), 'Threads left:', tc)

    for h in range(1,6):
        for i in range(3):
            t = threading.Thread(target=time_h_mlp, args=[h])
            t.start()
            tc[0]+=1