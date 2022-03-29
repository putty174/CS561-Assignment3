import csv
import sys


def load_files():
    for filename in sys.argv[1:]:
        with open(filename, 'r') as file:
            print(filename)
            cvs_reader = csv.reader(file)
            lines = []
            for line in cvs_reader:
                lines.append(line)
            print(lines)


def main():
    load_files()


if __name__ == "__main__":
    main()
