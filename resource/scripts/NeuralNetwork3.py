import math
import random
import sys
import time

import numpy
import scipy.special


def load_files():
    print("Loading Files...")
    training_images = load_file(sys.argv[1])
    training_labels = load_file(sys.argv[2])
    testing_images = load_file(sys.argv[3])
    print("Zipping Files...")
    return training_images, training_labels, testing_images


def load_file(filename):
    print("Loading {}".format(filename))
    with open(filename, 'r') as file:
        return numpy.genfromtxt(filename, dtype=int, delimiter=',')


class HandwritingDigitAnalysis:
    epochs = 35
    batch_size = 1000
    current_epoch = 0

    input_to_h1_reduction_ratio = 0.5
    h1_to_h2_reduction_ratio = 0.5

    error_adj_weight = 0.1

    output_classes = range(10)

    def __init__(self, training_labels, training_images, desired_output_size):
        self.start_time = time.time()

        self.input_size = len(training_images[0])
        self.output_size = desired_output_size

        self.hidden1_size = round(math.sqrt(self.input_size) * self.input_to_h1_reduction_ratio)
        self.hidden1_matrix = numpy.full((self.input_size, pow(self.hidden1_size, 2)), self.generate_initial_weights(self.input_size, pow(self.hidden1_size, 2)))

        self.hidden2_size = round((self.hidden1_size + self.output_size) / 2)
        self.hidden2_matrix = numpy.full((pow(self.hidden1_size, 2), pow(self.hidden2_size, 2)), self.generate_initial_weights(pow(self.hidden1_size, 2), pow(self.hidden2_size, 2)))

        self.output_matrix = numpy.full((pow(self.hidden2_size, 2), self.output_size), self.generate_initial_weights(pow(self.hidden2_size, 2), self.output_size))

        training_set = list(zip(training_labels, training_images))

        for i in range(self.epochs):
            self.current_epoch += 1
            print("{} Training epoch {}".format(time.time() - self.start_time, i))
            random.shuffle(training_set)
            accuracy = self.train(training_set)
            print("Epoch {}: {}%".format(i, accuracy))

    def generate_initial_weights(self, width, height):
        return numpy.random.uniform(-1., 1., size=(width, height)) / numpy.sqrt(width * height)

    def create_batches(self, source_labels, source_images, target_batch_size):
        zipped = list(zip(source_labels, source_images))
        return random.shuffle(zipped)

    def train(self, training_set):
        count = 0
        correct = 0
        cumulative_layer1_data_change = numpy.zeros_like(self.hidden1_matrix)
        cumulative_layer2_data_change = numpy.zeros_like(self.hidden2_matrix)
        cumulative_output_data_change = numpy.zeros_like(self.output_matrix)
        for label, data in training_set:
            count += 1

            if count % self.batch_size == 0:
                print("{}".format(cumulative_output_data_change[0]))
                self.update_model(cumulative_layer1_data_change, cumulative_layer2_data_change, cumulative_output_data_change)
                cumulative_layer1_data_change = numpy.zeros_like(self.hidden1_matrix)
                cumulative_layer2_data_change = numpy.zeros_like(self.hidden2_matrix)
                cumulative_output_data_change = numpy.zeros_like(self.output_matrix)
                print("{} trained".format(count))
                print("{}".format(self.output_matrix[0]))

            layer1 = data.dot(self.hidden1_matrix)
            sig1 = scipy.special.expit(layer1)
            layer2 = sig1.dot(self.hidden2_matrix)
            sig2 = scipy.special.expit(layer2)
            out_raw = sig2.dot(self.output_matrix)
            out_prob = scipy.special.softmax(out_raw)
            out_index = numpy.argmax(out_prob)
            out_result = self.output_classes[out_index]

            if out_result == label:
                correct += 1

            target = numpy.zeros(len(self.output_classes))
            target[label] = 1

            output_loss = self.log_loss(out_prob, target)
            output_adjust = output_loss * self.d_softmax(out_raw)
            output_matrix_delta = numpy.matmul(sig2.reshape(-1, 1), output_adjust.reshape(1, -1))
            if math.isnan(output_matrix_delta[0][0]):
                print("Problem with output adjust")
            cumulative_output_data_change += output_matrix_delta

            layer2_adjust = (self.output_matrix.dot(output_adjust.reshape(-1, 1))).T * self.d_sigmoid(sig2)
            hidden2_matrix_delta = numpy.matmul(sig1.reshape(-1, 1), layer2_adjust.reshape(1, -1))
            if math.isnan(hidden2_matrix_delta[0][0]):
                print("Problem with layer2 adjust")
            cumulative_layer2_data_change += hidden2_matrix_delta

            layer1_adjust = (self.hidden2_matrix.dot(layer2_adjust.reshape(-1, 1))).T * self.d_sigmoid(sig1)
            hidden1_matrix_delta = numpy.matmul(data.reshape(-1, 1), layer1_adjust.reshape(1, -1))
            if math.isnan(hidden1_matrix_delta[0][0]):
                print("Problem with layer1 adjust")
            cumulative_layer1_data_change += hidden1_matrix_delta

        return correct / len(training_set)

    def d_sigmoid(self, post_sigmoid_input):
        return post_sigmoid_input * (1 - post_sigmoid_input)

    def d_softmax(self, pre_softmax_input):
        exp = numpy.exp(pre_softmax_input - pre_softmax_input.max())
        return exp / numpy.sum(exp) * (1 - exp / numpy.sum(exp))

    def log_loss(self, output, target):
        output[output < numpy.finfo(numpy.float64).eps] = numpy.finfo(numpy.float64).eps

        return -numpy.sum(target * numpy.log(output)) / output.shape[0]

    def analyze_error(self, my_result, expected_result):
        error = expected_result - my_result
        loss = error * self.d_sigmoid()

    def update_model(self, hidden1_update, hidden2_update, output_update):

        adjustment_weight = ((self.epochs - self.current_epoch) / self.epochs) * self.error_adj_weight

        self.hidden1_matrix = self.hidden1_matrix - adjustment_weight * hidden1_update
        self.hidden2_matrix = self.hidden2_matrix - adjustment_weight * hidden2_update
        self.output_matrix = self.output_matrix - adjustment_weight * output_update

    def test(self, testing_set):
        print("testing")
        return self.output_size


def main():
    training_images, training_labels, testing_images = load_files()
    hda = HandwritingDigitAnalysis(training_labels, training_images, 10)


if __name__ == "__main__":
    main()
