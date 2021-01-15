#!/usr/bin/python3

import math
import random

class Connection:
    def __init__(self):
        self.weight = None
        self.left_neuron = None
        self.right_neuron = None

class Neuron:
    def __init__(self):
        self.output = None
        self.connections = []

    # Input function - sum of the products of weights and neuron state
    def calculate_output(self):
        result = 0

        for con in self.connections:
            neuron = con.left_neuron

            con_weight = con.weight
            neuron_output = neuron.output

            result = result + (con_weight * neuron_output)

        self.output = self.g(result)

    # Activation function
    def g(self, x):
        return self.sigmoid(x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

class NeuralNetwork():
    def __init__(self):
        self.trained = False
        self.input_layer = []
        # self.hidden_layer = []
        self.output_neuron = None
        # self.rand_w_multiplier = 1
        # self.learning_rate = 0.9
        # self.momentum = 0.7

        # Logical AND
        self.inputs = [ [1, 1], [1, 0], [0, 1], [0, 0] ]
        self.exp_outputs = [ 1, 0, 0, 0 ]
        self.result_outputs = []

    def create_neurons(self, value_one, value_two):
        # Create input neurons
        neuron_input_one = Neuron()
        neuron_input_one.output = value_one

        neuron_input_two = Neuron()
        neuron_input_two.output = value_two

        # Add them to the input layer
        self.input_layer.append(neuron_input_one)
        self.input_layer.append(neuron_input_two)

        # Create output neuron
        neuron_output = Neuron()
        self.output_neuron = neuron_output

        # Add bias
        bias_neuron = Neuron()
        bias_neuron.output = 1
        neuron_bias_con = Connection()

        random_value = random.uniform(-0.5, 0.5)
        neuron_bias_con.weight = random_value

        # Add bias connection
        neuron_bias_con.left_neuron =  bias_neuron
        neuron_bias_con.right_neuron = neuron_output
        neuron_output.connections.append(neuron_bias_con)

        # Add connections - input - output layer
        connection_one = Connection()
        connection_one.left_neuron = neuron_input_one
        connection_one.right_neuron = neuron_output

        random_value = random.uniform(-0.5, 0.5)
        connection_one.weight = random_value
        neuron_output.connections.append(connection_one)

        connection_two = Connection()
        connection_two.left_neuron = neuron_input_two
        connection_two.right_neuron = neuron_output

        random_value = random.uniform(-0.5, 0.5)
        connection_two.weight = random_value
        neuron_output.connections.append(connection_two)

    def set_training_inputs(self, inputs_list):
        self.inputs = inputs_list

    def set_exp_outputs(self, outputs_list):
        self.exp_outputs = outputs_list

    def input(self, value_one, value_two):
        self.input_layer[0].output = value_one
        self.input_layer[1].output = value_two

    def train(self, inputs_list, outputs_list):
        self.set_training_inputs(inputs_list)
        self.set_exp_outputs(outputs_list)
        self.create_neurons(0, 0)
        iter = 0

        while iter <= 100000:
            iter = iter + 1

            for i in range(0, len(self.inputs)):
                self.input(self.inputs[i][0], self.inputs[i][1])

                self.output_neuron.calculate_output()
                output = self.output_neuron.output
                expected_output = self.exp_outputs[i]

                error = output * (1 - output) * (expected_output - output)

                for con in self.output_neuron.connections:
                    out = con.left_neuron.output
                    con.weight = con.weight + 0.7 * error * out

    def test(self):
        final_values = []

        for i in range(0, len(self.inputs)):
            input_one = self.inputs[i][0]
            input_two = self.inputs[i][1]

            self.input(input_one, input_two)
            self.output_neuron.calculate_output()
            self.result_outputs.append(self.output_neuron.output)
            final_values.append(round(self.output_neuron.output))

        for i in range(0, len(self.inputs)):
            print('Input: {0} '.format(self.inputs[i]), end = '')
            print('Output: {0} '.format(final_values[i]), end = '')
            print('Percentages: {0}'.format(self.result_outputs[i]))

    def manual_test(self, x1_val, x2_val):
        self.input(x1_val, x2_val)
        self.output_neuron.calculate_output()
        output = self.output_neuron.output

        print('Input: [{0}, {1}] '.format(x1_val, x2_val), end = '')
        print('Output: {0} '.format(round(output)), end = '')
        print('Percentages: {0}'.format(output))

if __name__ == '__main__':
    inputs_and = [ [1, 1], [1, 0], [0, 1], [0, 0] ]
    exp_outputs_and = [ 1, 0, 0, 0 ]

    inputs_or = [ [1, 1], [1, 0], [0, 1], [0, 0] ]
    exp_outputs_or = [ 1, 1, 1, 0 ]

    print('\t\t    LOGICAL AND')
    neural_network_and = NeuralNetwork()
    neural_network_and.train(inputs_and, exp_outputs_and)
    neural_network_and.test()
    print()

    print('\t\t     LOGICAL OR')
    neural_network_or = NeuralNetwork()
    neural_network_or.train(inputs_or, exp_outputs_or)
    neural_network_or.test()
    print()

    # # For manual test:
    # print('Manual test AND')
    # x1_val = int(input())
    # x2_val = int(input())
    # neural_network_and.manual_test(x1_val, x2_val)
    # print()
    #
    # # For manual test:
    # print('Manual test OR')
    # x1_val = int(input())
    # x2_val = int(input())
    # neural_network_or.manual_test(x1_val, x2_val)
    # print()
