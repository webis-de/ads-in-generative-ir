import argparse
import pandas as pd
import os
import json

from ads_in_generative_ir import PROJECT_PATH

DATA_PATH = PROJECT_PATH / "data"

def get_judgements(file_name):
    df = pd.read_csv(DATA_PATH / f"{file_name}.csv")

    try:
        with open(DATA_PATH / f"{file_name}.json", "r") as json_file:
            mapping = json.load(json_file)
    except:
        mapping = {}

    for i, row in df.iterrows():
        if row["id"] in mapping:
            continue

        print(f"Query: {row['query']}")
        print(f"\nResponse:\n{row['response']}\n")
        answer = input("Ad (y/n): ")
        mapping[row["id"]] = 1 if answer.lower() == "y" else 0
        os.system('clear')

        with open(DATA_PATH / f"{file_name}.json", "w") as outfile:
            json.dump(mapping, outfile)

    df["ad"] = df["id"].apply(lambda id_: mapping[id_])
    df.to_csv(DATA_PATH / f"{file_name}_labeled.csv")

if __name__ == "__main__":
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    parser = argparse.ArgumentParser(
        prog='Manual Labeling',
        description='Annotate responses for having an ad or not')

    parser.add_argument('-f', '--file_name', type=str, default='gpt4_detections',
                        help='File suffix to load responses from.')
    args = parser.parse_args()
    get_judgements(args.file_name)
