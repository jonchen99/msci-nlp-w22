import sys

def main():
    print("Hello World")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("ERROR: Invalid number of inputs")
        exit(1)

    input_folder_path = sys.argv[1]
    print("Input folder is " + input_folder_path)
    main()