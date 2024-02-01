from collections import Counter
from os.path import isfile
from openai import OpenAI
from pandas import read_csv
import re
from tqdm import tqdm

from ads_in_generative_ir import RESOURCE_PATH
from ads_in_generative_ir.generate.utils import prompt_gpt

class AdvertisementSelector:
    def __init__(self, meta_topic: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.meta_topic = meta_topic
        self.out_file = RESOURCE_PATH / f'advertisements/{meta_topic}_query_pairs.csv'

        response_df = read_csv(RESOURCE_PATH / f'generated_responses/{meta_topic}_filtered.csv')

        # Load the queries and discard any for which ads were already selected
        try:
            pair_df = read_csv(self.out_file)
            covered_queries = pair_df["query"].unique().tolist()
        except:
            covered_queries = []
        candidate_queries = response_df["query"].unique().tolist()
        self.queries = [q for q in candidate_queries if q not in covered_queries]

        # Prepare the candidates
        quality_df = read_csv(RESOURCE_PATH / f'advertisements/{meta_topic}_qualities.csv')
        self.candidate_ads = quality_df["advertisement"].tolist()
        self.brands = quality_df.loc[quality_df.type == "brand"].apply(
            lambda x: f"{x['advertisement']} [{x['type']}, Qualities: {x['qualities']}]", axis=1).to_list()
        self.products = quality_df.loc[quality_df.type != "brand"].apply(
            lambda x: f"{x['advertisement']} [{x['type']}, Qualities: {x['qualities']}]", axis=1).to_list()

        # Write the header if it does not exist
        if not isfile(self.out_file):
            with open(self.out_file, 'w') as file:
                file.write("query,advertisement\n")

    def select_advertisements(self, max_ads: int = None):
        max_ads = max_ads or 5
        base_prompt = f"Take the following keyword query to a search engine and select at least 2 and at most {max_ads} " \
                      "suitable products or brands from the provided list to be advertised on the result page. " \
                      "Please return only the selected elements, separated by '***'.\n\n" \
                      f"Products: {', '.join(self.products)}\n" \
                      f"Brands: {', '.join(self.brands)}\n"

        with open(self.out_file, 'a') as file:
            for query in tqdm(self.queries, desc=f"Selecting products/brands for {self.meta_topic} queries"):
                text = prompt_gpt(self.client, prompt=base_prompt + f"Query: {query}")
                for advertisement in text.split("***"):
                    cleaned_ad = re.sub(" \[.*?\]", "", advertisement.strip())
                    file.write(f"{query},{cleaned_ad}\n")

        # Clean the results by keeping only valid ads
        pair_df = read_csv(self.out_file)
        pair_df = pair_df.loc[pair_df["advertisement"].isin(self.candidate_ads)]
        pair_df.to_csv(self.out_file)


def check_ad_coverage(meta_topic: str, pair_df_suffix: str = "query_pairs"):
    """
    Check which ads are not covered by the selected responses
    """
    pair_df = read_csv(RESOURCE_PATH / f'advertisements/{meta_topic}_{pair_df_suffix}.csv')
    quality_df = read_csv(RESOURCE_PATH / f'advertisements/{meta_topic}_qualities.csv')

    candidate_ads = quality_df[["advertisement", "qualities"]].values.tolist()
    covered_ads = pair_df["advertisement"].tolist()
    counts = Counter(covered_ads).most_common()
    most_common = counts[:10]
    least_common = counts[-10:]

    print(f"\nThe following ads were selected for none of the {meta_topic} queries:")
    for tup in candidate_ads:
        if tup[0] not in covered_ads:
            print("- {0:40}  ({1})".format(tup[0], tup[1]))

    print(f"\nThe following ads were selected the most for {meta_topic} queries:")
    for tup in most_common:
        print("- {0:40}  {1}".format(tup[0], tup[1]))

    print(f"\nThe following ads were selected the least for {meta_topic} queries:")
    for tup in least_common:
        ad = tup[0]
        try:
            quality = quality_df.loc[quality_df["advertisement"] == ad, "qualities"].values[0]
        except Exception as e:
            print(f"Failed exception with {e} for {ad}")
        print("- {0:40}  {1} ({2})".format(ad, tup[1], quality))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='Advertisement Selection',
        description='Send prompts to GPT-4 to select suitable products or brands for a given query')

    parser.add_argument("meta_topic", metavar="M", type=str, help="Meta topic for which to select advertisements")
    parser.add_argument("key", metavar="K", type=str, help="Key for the OpenAI API")
    parser.add_argument('-a', '--ads', type=str, help="Number of ads to select at most per query (default 5)")
    args = parser.parse_args()

    selector = AdvertisementSelector(meta_topic=args.meta_topic, api_key=args.key)
    selector.select_advertisements(max_ads=args.ads)