#!/usr/bin/python3

import math
import random

class Connection:
    def __init__(self):
        self.type = None
        self.weight = None
        self.left_neuron = None
        self.right_neuron = None
        self.prev_delta_weight = 0
        self.delta_weight = 0

class Neuron:
    def __init__(self):
        self.type = None
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

    def calculate_output_xor(self):
        # print('called')
        result = 0

        for con in self.connections:
            if (self.type == 'HIDDEN' and con.right_neuron.type != 'OUTPUT') or self.type == 'OUTPUT':
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
        self.hidden_layer = []
        self.output_neuron = None

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

    def create_neurons_xor(self, value_one, value_two):
        # Create input neurons
        neuron_input_one = Neuron()
        neuron_input_one.type = 'INPUT'
        neuron_input_one.output = value_one

        neuron_input_two = Neuron()
        neuron_input_two.type = 'INPUT'
        neuron_input_two.output = value_two

        # Add them to the input layer
        self.input_layer.append(neuron_input_one)
        self.input_layer.append(neuron_input_two)

        # Create hidden layer
        neuron_hidden_one = Neuron()
        neuron_hidden_one.type = 'HIDDEN'
        neuron_hidden_two = Neuron()
        neuron_hidden_two.type = 'HIDDEN'

        self.hidden_layer.append(neuron_hidden_one)
        self.hidden_layer.append(neuron_hidden_two)

        # Add bias for the hidden layer
        bias_neuron_one = Neuron()
        bias_neuron_one.type = 'BIAS'
        bias_neuron_one.output = 1

        # Bias - hidden neuron one
        neuron_bias_con_one = Connection()
        neuron_bias_con_one.type = 'BIAS-HIDDEN'
        random_value = random.uniform(-0.5, 0.5)
        neuron_bias_con_one.weight = random_value

        neuron_bias_con_one.left_neuron = bias_neuron_one
        neuron_bias_con_one.right_neuron = neuron_hidden_one
        neuron_hidden_one.connections.append(neuron_bias_con_one)
        bias_neuron_one.connections.append(neuron_bias_con_one)

        # Bias - hidden neuron two
        neuron_bias_con_two = Connection()
        neuron_bias_con_two.type = 'BIAS-HIDDEN'
        random_value = random.uniform(-0.5, 0.5)
        neuron_bias_con_two.weight = random_value

        neuron_bias_con_two.left_neuron = bias_neuron_one
        neuron_bias_con_two.right_neuron = neuron_hidden_two
        neuron_hidden_two.connections.append(neuron_bias_con_two)
        bias_neuron_one.connections.append(neuron_bias_con_two)

        # Create output neuron
        neuron_output = Neuron()
        neuron_output.type = 'OUTPUT'
        self.output_neuron = neuron_output

        # Add bias to output
        bias_neuron = Neuron()
        bias_neuron.type = 'BIAS'
        bias_neuron.output = 1

        # Bias - output
        neuron_bias_con = Connection()
        neuron_bias_con.type = 'BIAS-OUTPUT'
        random_value = random.uniform(-0.5, 0.5)
        neuron_bias_con.weight = random_value

        neuron_bias_con.left_neuron = bias_neuron
        neuron_bias_con.right_neuron = neuron_output
        neuron_output.connections.append(neuron_bias_con)
        bias_neuron.connections.append(neuron_bias_con)

        # Connect inputs with hidden layer
        con_one = Connection()
        con_one.type = 'INPUT-HIDDEN'
        random_value = random.uniform(-0.5, 0.5)
        con_one.weight = random_value
        con_one.left_neuron = neuron_input_one
        con_one.right_neuron = neuron_hidden_one
        neuron_hidden_one.connections.append(con_one)
        neuron_input_one.connections.append(con_one)

        con_two = Connection()
        con_two.type = 'INPUT-HIDDEN'
        random_value = random.uniform(-0.5, 0.5)
        con_two.weight = random_value
        con_two.left_neuron = neuron_input_two
        con_two.right_neuron = neuron_hidden_one
        neuron_hidden_one.connections.append(con_two)
        neuron_input_two.connections.append(con_two)

        con_three = Connection()
        con_three.type = 'INPUT-HIDDEN'
        random_value = random.uniform(-0.5, 0.5)
        con_three.weight = random_value
        con_three.left_neuron = neuron_input_one
        con_three.right_neuron = neuron_hidden_two
        neuron_hidden_two.connections.append(con_three)
        neuron_input_one.connections.append(con_three)

        con_four = Connection()
        con_four.type = 'INPUT-HIDDEN'
        random_value = random.uniform(-0.5, 0.5)
        con_four.weight = random_value
        con_four.left_neuron = neuron_input_two
        con_four.right_neuron = neuron_hidden_two
        neuron_hidden_two.connections.append(con_four)
        neuron_input_two.connections.append(con_four)

        # Connect hidden layer to output
        con_one_output = Connection()
        con_one_output.type = 'HIDDEN-OUTPUT'
        random_value = random.uniform(-0.5, 0.5)
        con_one_output.weight = random_value
        con_one_output.left_neuron = neuron_hidden_one
        con_one_output.right_neuron = neuron_output
        neuron_output.connections.append(con_one_output)
        neuron_hidden_one.connections.append(con_one_output)

        con_two_output = Connection()
        con_two_output.type = 'HIDDEN-OUTPUT'
        random_value = random.uniform(-0.5, 0.5)
        con_two_output.weight = random_value
        con_two_output.left_neuron = neuron_hidden_two
        con_two_output.right_neuron = neuron_output
        neuron_output.connections.append(con_two_output)
        neuron_hidden_two.connections.append(con_two_output)

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

    def forward_xor(self):
        for con in self.output_neuron.connections:
            if con.left_neuron.type == 'HIDDEN':
                con.left_neuron.calculate_output_xor()

        self.output_neuron.calculate_output_xor()

    def backward_xor(self, i):
        output = self.output_neuron.output
        expected_output = self.exp_outputs[i]

        error_output = output * (1 - output) * (expected_output - output)

        # Fix the weights of the connections to the output neuron
        for con in self.output_neuron.connections:
            output_con = con.left_neuron.output
            con.weight = con.weight + 0.25 * error_output * output_con

        for con in self.output_neuron.connections:
            cur_neuron = con.left_neuron

            if cur_neuron.type == 'HIDDEN':
                hidden_neuron_out = cur_neuron.output

                sum = 0
                for cur_neuron_con in cur_neuron.connections:
                    if cur_neuron_con.right_neuron.type == 'OUTPUT':
                        sum = sum + (cur_neuron_con.weight * error_output)

                hidden_layer_error = hidden_neuron_out * (1 - hidden_neuron_out) * sum

                for cur_neuron_con in cur_neuron.connections:
                    if cur_neuron_con.right_neuron.type != 'OUTPUT':
                        input_neuron_out = cur_neuron_con.left_neuron.output
                        cur_neuron_con.weight = cur_neuron_con.weight + 0.25 * hidden_layer_error * input_neuron_out

    def train_xor(self, inputs_list, outputs_list):
        self.set_training_inputs(inputs_list)
        self.set_exp_outputs(outputs_list)
        self.create_neurons_xor(0, 0)
        iter = 0

        while iter <= 100000:
            iter = iter + 1

            for i in range(0, len(self.inputs)):
                self.input(self.inputs[i][0], self.inputs[i][1])

                self.forward_xor()
                self.backward_xor(i)

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
            print('Expected: {0} '.format(self.exp_outputs[i]), end = '')
            print('Percentages: {0}'.format(self.result_outputs[i]))

    def test_xor(self):
        final_values = []

        for i in range(0, len(self.inputs)):
            input_one = self.inputs[i][0]
            input_two = self.inputs[i][1]

            self.input(input_one, input_two)
            self.forward_xor()
            self.result_outputs.append(self.output_neuron.output)
            final_values.append(round(self.output_neuron.output))

        for i in range(0, len(self.inputs)):
            print('Input: {0} '.format(self.inputs[i]), end = '')
            print('Output: {0} '.format(final_values[i]), end = '')
            print('Expected: {0} '.format(self.exp_outputs[i]), end = '')
            print('Percentages: {0}'.format(self.result_outputs[i]))

    def manual_test(self, x1_val, x2_val):
        self.input(x1_val, x2_val)
        self.output_neuron.calculate_output()
        output = self.output_neuron.output

        print('Input: [{0}, {1}] '.format(x1_val, x2_val), end = '')
        print('Output: {0} '.format(round(output)), end = '')
        print('Percentages: {0}'.format(output))

    def manual_test_xor(self, x1_val, x2_val):
        self.input(x1_val, x2_val)
        self.forward_xor()
        output = self.output_neuron.output

        print('Input: [{0}, {1}] '.format(x1_val, x2_val), end = '')
        print('Output: {0} '.format(round(output)), end = '')
        print('Percentages: {0}'.format(output))

if __name__ == '__main__':
    random.seed(3)

    inputs_and = [ [1, 1], [1, 0], [0, 1], [0, 0] ]
    exp_outputs_and = [ 1, 0, 0, 0 ]

    inputs_or = [ [1, 1], [1, 0], [0, 1], [0, 0] ]
    exp_outputs_or = [ 1, 1, 1, 0 ]

    inputs_nand = [ [1, 1], [1, 0], [0, 1], [0, 0] ]
    exp_outputs_nand = [ 0, 1, 1, 1 ]

    inputs_xor = [ [1, 1], [1, 0], [0, 1], [0, 0] ]
    exp_outputs_xor = [ 0, 1, 1, 0 ]

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

    print('\t\t    LOGICAL XOR')
    neural_network_xor = NeuralNetwork()
    neural_network_xor.train_xor(inputs_xor, exp_outputs_xor)
    neural_network_xor.test_xor()
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
    #
    # # For manual test:
    # print('Manual test XOR')
    # x1_val = int(input())
    # x2_val = int(input())
    # neural_network_xor.manual_test(x1_val, x2_val)
    # print()
