import math
import random
import sys
import time
from enum import Enum
from idlelib import testing

import numpy
import scipy.special


def load_files():
    print("Loading Files...")
    training_images = load_file(sys.argv[1])
    training_images = preprocess_images(training_images)
    training_labels = load_file(sys.argv[2])
    testing_images = load_file(sys.argv[3])
    testing_images = preprocess_images(testing_images)
    print("Zipping Files...")
    return training_images, training_labels, testing_images


def load_file(filename):
    print("Loading {}".format(filename))
    with open(filename, 'r') as file:
        return numpy.genfromtxt(filename, dtype=int, delimiter=',')


def preprocess_images(images):
    return images / numpy.amax(images)

class ActivationFunction(Enum):
    sigmoid = "sigmoid"
    relu = "relu"

class HandwritingDigitAnalysis:
    epochs = 40
    batch_size = 100
    current_epoch = 0

    input_to_h1_reduction_ratio = 0.5

    error_adj_weight = 0.01

    output_classes = range(10)

    def __init__(self, training_labels, training_images, desired_output_size):
        self.activation_type = ActivationFunction.relu
        self.start_time = time.time()

        self.input_size = len(training_images[0])
        self.output_size = desired_output_size

        self.hidden1_size = round(math.sqrt(self.input_size) * self.input_to_h1_reduction_ratio)
        self.hidden1_matrix = numpy.full((self.input_size, pow(self.hidden1_size, 2)), self.generate_initial_weights(self.input_size, pow(self.hidden1_size, 2)))
        self.hidden1_bias = numpy.full((1, pow(self.hidden1_size, 2)), self.generate_initial_weights(1, pow(self.hidden1_size, 2)))

        self.output_matrix = numpy.full((pow(self.hidden1_size, 2), self.output_size), self.generate_initial_weights(pow(self.hidden1_size, 2), self.output_size))
        self.output_bias = numpy.full((1, self.output_size), self.generate_initial_weights(1, self.output_size))

        training_set = list(zip(training_labels, training_images))

        for i in range(self.epochs):
            self.current_epoch += 1
            print("Training epoch {}".format(i))
            random.shuffle(training_set)
            accuracy = self.train(training_set)
            print("{} Epoch {}: {}%".format(time.time() - self.start_time, i, accuracy))

    def generate_initial_weights(self, width, height):
        return numpy.random.uniform(-1., 1., size=(width, height)) / numpy.sqrt(width * height)

    def train(self, training_set):
        count = 0
        correct = 0
        cumulative_layer1_data_change = numpy.zeros_like(self.hidden1_matrix)
        cumulative_layer1_bias_change = numpy.zeros_like(self.hidden1_bias)
        cumulative_output_data_change = numpy.zeros_like(self.output_matrix)
        cumulative_output_bias_change = numpy.zeros_like(self.output_bias)

        for label, data in training_set:
            count += 1
            data = data.reshape(1, -1)

            if count % self.batch_size == 0:
                # print("{}".format(cumulative_output_data_change[0]))
                self.update_model(cumulative_layer1_data_change, cumulative_layer1_bias_change, cumulative_output_data_change, cumulative_output_bias_change)
                cumulative_layer1_data_change = numpy.zeros_like(self.hidden1_matrix)
                cumulative_layer1_bias_change = numpy.zeros_like(self.hidden1_bias)
                cumulative_output_data_change = numpy.zeros_like(self.output_matrix)
                cumulative_output_bias_change = numpy.zeros_like(self.output_bias)
                if count % (self.batch_size * 100) == 0:
                    print("{} trained".format(count))
                # print("{}".format(self.output_matrix[0]))

            layer1 = data.dot(self.hidden1_matrix)
            layer1 += self.hidden1_bias
            activated1 = self.activation(layer1)

            out_raw = activated1.dot(self.output_matrix)
            out_raw += self.output_bias
            out_prob = scipy.special.softmax(out_raw)
            out_index = numpy.argmax(out_prob)
            out_result = self.output_classes[out_index]

            if out_result == label:
                correct += 1

            target = numpy.zeros(len(self.output_classes))
            target[label] = 1

            error = out_prob - target
            output_loss = self.log_loss(out_prob, target)

            cost = error / self.batch_size
            cumulative_output_bias_change += cost
            output_matrix_delta = numpy.matmul(activated1.T, cost)
            cumulative_output_data_change += output_matrix_delta

            layer1_bias_delta = numpy.matmul(cost, self.output_matrix.T) * self.d_activation(layer1)
            cumulative_layer1_bias_change += layer1_bias_delta
            layer1_matrix_delta = numpy.matmul(data.T, layer1_bias_delta)
            cumulative_layer1_data_change += layer1_matrix_delta

        return correct / len(training_set)

    def d_sigmoid(self, d_input):
        sigmoid = scipy.special.expit(d_input)
        return sigmoid * (1 - sigmoid)

    def relu(self, f_input):
        return numpy.maximum(0, f_input)

    def d_relu(self, d_input):
        return numpy.where(d_input > 0, 1, 0)

    def activation(self, a_input):
        if self.activation_type == ActivationFunction.sigmoid:
            return scipy.special.expit(a_input)
        elif self.activation_type == ActivationFunction.relu:
            return self.relu(a_input)

    def d_activation(self, dev_input):
        if self.activation_type == ActivationFunction.sigmoid:
            return self.d_sigmoid(dev_input)
        elif self.activation_type == ActivationFunction.relu:
            return self.d_relu(dev_input)

    def d_softmax(self, pre_softmax_input):
        exp = numpy.exp(pre_softmax_input - pre_softmax_input.max())
        return exp / numpy.sum(exp) * (1 - exp / numpy.sum(exp))

    def log_loss(self, output, target):
        total = 0
        for i in range(len(output)):
            total += output[i] * math.log(numpy.finfo(numpy.float64).eps + target[i])
        return -total / len(output)

    def analyze_error(self, my_result, expected_result):
        error = expected_result - my_result
        loss = error * self.d_sigmoid()

    def update_model(self, hidden1_update, hidden1_bias_update, output_update, output_bias_update):

        adjustment_weight = ((self.epochs - self.current_epoch) / self.epochs) * self.error_adj_weight

        self.hidden1_matrix = self.hidden1_matrix - adjustment_weight * hidden1_update
        self.output_matrix = self.output_matrix - adjustment_weight * output_update

        self.hidden1_bias = self.hidden1_bias - (adjustment_weight * hidden1_bias_update)
        self.output_bias = self.output_bias - (adjustment_weight * output_bias_update)

    def test(self, testing_set):
        print("testing")
        return self.output_size


def main():
    training_images, training_labels, testing_images = load_files()
    hda = HandwritingDigitAnalysis(training_labels, training_images, 10)


if __name__ == "__main__":
    main()
