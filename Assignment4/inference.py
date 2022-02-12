# Jonathan Chen, 20722167
# University of Waterloo
# March 4, 2022

import sys

def main(text_file):
    print("Hello World")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: Invalid number of inputs")
        exit(1)

    input_text_file = sys.argv[1]
    classifier = sys.argv[2]

    main(input_text_file)
