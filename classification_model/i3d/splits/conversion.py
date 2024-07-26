import csv
import os

def process_file(file_name):
    try:
        # Read the original content
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            lines = []
            for row in reader:
                try:
                    name, number = row
                    # Convert number to integer if it is a float
                    number = int(float(number))
                    lines.append((name, number))
                except ValueError:
                    print(f"Skipping line in {file_name} due to conversion error: {row}")

        # Write the modified content back to the same file
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(lines)

    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_name}: {e}")

# Directory containing the files
directory = "/Users/Jivan/MM_Retrieval_App/classification_model/i3d/splits/"

# List of files to process
files = ["test_CV2.csv", "train_CV2.csv", "validation_CV2.csv"]

# Process each file
for file_name in files:
    file_path = os.path.join(directory, file_name)
    process_file(file_path)

print("Files have been processed and updated.")