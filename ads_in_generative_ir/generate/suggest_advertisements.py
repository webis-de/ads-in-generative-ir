from openai import OpenAI
from pandas import DataFrame

from ads_in_generative_ir import RESOURCE_PATH
from ads_in_generative_ir.generate.utils import prompt_gpt

class AdvertismentSuggestor:
    def __init__(self, meta_topic: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.meta_topic = meta_topic

        self.out_file = RESOURCE_PATH / f'advertisements/{meta_topic}_qualities.csv'

    def suggest(self, num_ads: int = 500):
        prompt = f"Please suggest {num_ads} products and/or brands to be advertised in queries" \
                 f"with the topic {self.meta_topic}. Notes: \n" \
                 "- Place 'brand' or the kind of product behind each suggestion \n" \
                 "- Add qualities of the brand or product to be advertised \n" \
                 "- Separate individual suggestions with '***' and product,type, and qualities with a semicolon \n\n" \
                 "For example:  \n" \
                 "McDonald's;brand;service,value\n" \
                 "***\n" \
                 "BigMac;burger;big,beef,tasty"

        answer = prompt_gpt(self.client, prompt)
        triples = [tuple(x.replace("\n", "").split(';')) for x in answer.split("***")]
        records = []
        for row in triples:
            try:
                ad, ad_type, qualities = row
                ad_type = "brand" if ("brand" in ad_type or "company" in ad_type) else ad_type
                qualities = f"{qualities.strip().replace('.', '')}"
                records.append((ad, ad_type, qualities))

            except Exception as e:
                print(f'{row} failed with exception {e}')



        df = DataFrame.from_records(records, columns=["advertisement", "type", "qualities"])
        df.to_csv(self.out_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='Advertisement Suggestion',
        description='Send prompt to GPT-4 to suggest products or brands for a meta topic')

    parser.add_argument("meta_topic", metavar="M", type=str, help="Meta topic for which to suggest advertisements")
    parser.add_argument("key", metavar="K", type=str, help="Key for the OpenAI API")
    args = parser.parse_args()

    suggestor = AdvertismentSuggestor(meta_topic=args.meta_topic, api_key=args.key)
    suggestor.suggest()