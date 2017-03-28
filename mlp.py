import numpy as np

def f(net):
    return 1 / (1 + np.exp(-net))

def df_dnet(net):
    return f(net) * (1 - f(net))

class my_little_poney:
    def __init__(
            self,
            input_layer_neurons = 2,
            hidden_layer_neurons = 2,
            output_layer_neurons = 1,
            f_function = f,
            df_dnet_function = df_dnet
            ):
        self.layers = {}

        self.f_function = f_function
        self.df_dnet_function = df_dnet_function

        self.layers['hidden'] = np.random.uniform(
                -0.5, 0.5,
                 hidden_layer_neurons * (input_layer_neurons + 1)
        ).reshape(hidden_layer_neurons, input_layer_neurons + 1)

        self.layers['output'] = np.random.uniform(
                -0.5, 0.5,
                 output_layer_neurons * (hidden_layer_neurons + 1)
        ).reshape(output_layer_neurons, hidden_layer_neurons + 1)

    def forward(self, input_pattern):
        size_hidden_layer = self.layers['hidden'].shape[0]
        size_output_layer = self.layers['output'].shape[0]

        f_h = np.zeros(size_hidden_layer)
        df_h = np.zeros(size_hidden_layer)
        for j in range(0, size_hidden_layer):
            input_pattern_normalized = input_pattern + [1]
            net_h = np.dot(input_pattern_normalized, self.layers['hidden'][j])
            f_h[j] = self.f_function(net_h)
            df_h[j] = self.df_dnet_function(net_h)

        f_o = np.zeros(size_output_layer)
        df_o = np.zeros(size_output_layer)
        for j in range(0, size_output_layer):
            f_h_normalized = f_h + [1]
            net_o = np.dot(f_h_normalized, self.layers['output'][j])
            f_o[j] = self.f_function(net_o)
            df_o[j] = self.df_dnet_function(net_o)

        fwd = {}
        fwd['f_h'] = f_h
        fwd['f_o'] = f_o
        fwd['df_h'] = df_h
        fwd['df_o'] = df_o

        return fwd

    def backpropagation(self, X, Y, eta = 0.1, threshold = 1e-2):
        squared_error = 2 * threshold
        while squared_error > threshold:
            squared_error = 0
            for p in range(0, len(X)):
                x_p = np.array(X[p])
                y_p = np.array(Y[p])

                fwd = self.forward(x_p)
                o_p = fwd['f_o']
                delta_p = y_p - o_p

                squared_error = squared_error + (delta_p ** 2)

                delta_output = delta_p * fwd['df_o']

                w_length = self.layers['output'].shape[1] - 1
                delta_hidden = fwd['df_h'] * (
                        np.dot(delta_output, self.layers['output'][:,0:(w_length-1)])
                        )

                self.layers['output'] = self.layers['output'] + (
                        eta * (np.dot(delta_output, fwd['f_h']))
                        )

                self.layers['hidden'] = self.layers['hidden'] + (
                        eta * (np.dot(delta_hidden, (x_p + [1])))
                        )

            squared_error = squared_error / X.shape[0]
            print("Avg sqr err: " + str(squared_error))
        return self

# main
def main():
    dataset = [[0,0,0],
               [0,1,1],
               [1,0,1],
               [1,1,0]]

    #X = dataset[:,0:2]
    #Y = dataset[:,2]
    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,1,1,0]

    print("Inputs: " + str(X))

    mlp = my_little_poney(2, 2, 1)
    mlp.backpropagation(X, Y)

    for p in range(0,4):
        x_p = X[p]
        y_p = Y[p]

        fwd = mlp.forward(x_p)

        print(x_p)
        print(y_p)
        print(fwd['f_o'])

if __name__ == "__main__":
    main()
