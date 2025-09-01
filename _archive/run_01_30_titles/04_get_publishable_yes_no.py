import csv


def read_pipe_delimited_csv(file_path):
    """
    Read a pipe-delimited CSV file and extract the 7th element from each row which is the postability of being publishable.

    Args:
        file_path (str): Path to the pipe-delimited CSV file

    Returns:
        list: List of 4th elements from each row
    """
    postability_elements = []

    try:
        with open(file_path, "r", encoding="utf-8") as csvfile:
            # Read file line by line to handle pipe-delimited format
            for line in csvfile:
                # Strip any trailing newline and split by pipe
                elements = line.strip().split("|")

                # Check if the row has at least 7 elements
                if len(elements) >= 7:

                    postability_element = elements[6].strip()
                    postability_elements.append(postability_element)
                else:
                    print(f"Skipping row with insufficient elements: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return postability_elements


def main():
    # Replace with the actual path to your pipe-delimited CSV file
    file_path = "./04_article_writer_publishable.csv"

    # Read and print the 4th elements
    results = read_pipe_delimited_csv(file_path)

    print("4th Elements:")
    for idx, element in enumerate(results, 1):
        print(f"Row {idx}: {element}")

    print(f"\nTotal rows processed: {len(results)}")


if __name__ == "__main__":
    main()
