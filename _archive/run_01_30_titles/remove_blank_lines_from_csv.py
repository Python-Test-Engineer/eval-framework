import csv
import pandas as pd


# Method 1: Using basic file operations (simplest)
def remove_blank_lines_basic(input_file, output_file):
    """Remove blank lines using basic file operations"""
    with (
        open(input_file, "r", newline="", encoding="utf-8") as infile,
        open(output_file, "w", newline="", encoding="utf-8") as outfile,
    ):

        for line in infile:
            if line.strip():  # If line is not empty after stripping whitespace
                outfile.write(line)


if __name__ == "__main__":
    input_csv = "./src/case_study1/langgraph/datasets/run_01_30_titles/04_article_writer_publishable.csv"
    output_csv = "./src/case_study1/langgraph/datasets/run_01_30_titles/04_article_writer_publishable_clean.csv"

    # Example usage
    remove_blank_lines_basic(input_csv, output_csv)
    print(f"Blank lines removed. Output saved to {output_csv}.")
