import os
import glob
import csv
import re

def extract_english_question(file_path):
    """
    Extracts English question from lines like:
    # [en] Which works have been composed by Mozart?
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Match: # [en] ....
            match = re.match(r"#\s*\[en\]\s*(.+)", line)
            if match:
                return match.group(1).strip()

    return None  # if no English question found


def main():
    rq_files = glob.glob("*.rq")
    output_file = "questions.csv"

    rows = []

    for file in rq_files:
        question = extract_english_question(file)
        if question:
            rows.append([question, file])

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "file_source"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} questions to {output_file}")


if __name__ == "__main__":
    main()
