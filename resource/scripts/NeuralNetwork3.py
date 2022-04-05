import math
import sys

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
    epochs = 100000
    batch_size = 1000

    starting_weight = 0.001
    input_to_h1_reduction_ratio = 0.5
    h1_to_h2_reduction_ratio = 0.5

    error_adj_weight = 0.1

    output_classes = range(10)

    def __init__(self, training_labels, training_images, desired_output_size):
        self.input_size = len(training_images[0])
        self.output_size = desired_output_size

        self.hidden1_size = round(math.sqrt(self.input_size) * self.input_to_h1_reduction_ratio)
        self.hidden1_matrix = numpy.full((self.input_size, pow(self.hidden1_size, 2)), self.generate_initial_weights(self.input_size, pow(self.hidden1_size, 2)))

        self.hidden2_size = round((self.hidden1_size + self.output_size) / 2)
        self.hidden2_matrix = numpy.full((pow(self.hidden1_size, 2), pow(self.hidden2_size, 2)), self.generate_initial_weights(pow(self.hidden1_size, 2), pow(self.hidden2_size, 2)))

        self.output_matrix = numpy.full((pow(self.hidden2_size, 2), self.output_size), self.generate_initial_weights(pow(self.hidden2_size, 2), self.output_size))

        for i in range(self.epochs):
            print("Training epoch {}".format(i))
            batch_set = self.create_batch(training_labels, training_images, self.batch_size)
            accuracy = self.train(batch_set)
            print("Epoch {}: {}%".format(i, accuracy))

    def generate_initial_weights(self, width, height):
        return numpy.random.uniform(-1., 1., size=(width, height)) / numpy.sqrt(width * height)

    def create_batch(self, source_labels, source_images, target_batch_size):
        sample_indexes = numpy.random.randint(0, len(source_images), size=target_batch_size)
        batch_labels = source_labels[sample_indexes]
        batch_images = source_images[sample_indexes]
        return tuple(zip(batch_labels, batch_images))

    def train(self, training_set):
        count = 0
        correct = 0
        for label, data in training_set:
            count += 1
            if count % 10000 == 0:
                print("{} trained".format(count))

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

            layer2_adjust = (self.output_matrix.dot(output_adjust.reshape(-1, 1))).T * self.d_sigmoid(sig2)
            hidden2_matrix_delta = numpy.matmul(sig1.reshape(-1, 1), layer2_adjust.reshape(1, -1))

            layer1_adjust = (self.hidden2_matrix.dot(layer2_adjust.reshape(-1, 1))).T * self.d_sigmoid(sig1)
            hidden1_matrix_delta = numpy.matmul(data.reshape(-1, 1), layer1_adjust.reshape(1, -1))

            self.output_matrix = self.output_matrix - self.error_adj_weight * output_matrix_delta
            self.hidden2_matrix = self.hidden2_matrix - self.error_adj_weight * hidden2_matrix_delta
            self.hidden1_matrix = self.hidden1_matrix - self.error_adj_weight * hidden1_matrix_delta

        return correct / len(training_set)

    def d_sigmoid(self, post_sigmoid_input):
        return post_sigmoid_input * (1 - post_sigmoid_input)

    def d_softmax(self, pre_softmax_input):
        exp = numpy.exp(pre_softmax_input - pre_softmax_input.max())
        return exp / numpy.sum(exp) * (1 - exp / numpy.sum(exp))

    def log_loss(self, output, target):
        return -numpy.sum(target * numpy.log(output)) / output.shape[0]

    def analyze_error(self, my_result, expected_result):
        error = expected_result - my_result
        loss = error * self.d_sigmoid()

    def test(self, testing_set):
        print("testing")
        return self.output_size


def main():
    training_images, training_labels, testing_images = load_files()
    hda = HandwritingDigitAnalysis(training_labels, training_images, 10)


if __name__ == "__main__":
    main()
