import csv
import sys


def load_files():
    training_images = load_file(sys.argv[1])
    training_lables = load_file(sys.argv[2])
    testing_images = load_file(sys.argv[3])
    training_tuples = tuple(zip(training_lables, training_images))
    return training_tuples, testing_images


def load_file(filename):
    with open(filename, 'r') as file:
        cvs_reader = csv.reader(file)
        lines = []
        for line in cvs_reader:
            lines.append(line)
    return lines


def main():
    training_set, testing_set = load_files()
    print(training_set)


if __name__ == "__main__":
    main()
