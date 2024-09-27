import json
import argparse
from datasets import load_dataset

def save_to_json(data, output_path):
    with open(output_path, 'w+') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSON files in a folder")
    parser.add_argument("--input_path", type=str, help="Path to the folder containing JSON files")
    parser.add_argument("--output_path", type=str, help="Path to the folder containing JSON files")
    args = parser.parse_args()

    dataset = load_dataset(args.input_path, split="train")
    dataset = dataset.map(lambda sample: {"generator": args.input_path})

    save_to_json(dataset.to_list(), args.output_path)
    print(f"Merged data saved to {args.output_path}")
