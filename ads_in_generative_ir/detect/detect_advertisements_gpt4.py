from openai import OpenAI
import pandas as pd
from datasets import load_from_disk
from bleu import multi_list_bleu
import os.path

from ads_in_generative_ir import RESOURCE_PATH, PROJECT_PATH
from ads_in_generative_ir.generate.utils import prompt_gpt
DATA_PATH = PROJECT_PATH / "data"

'''
To run the ad detection with GPT-4, you need to specify an api_key and pass it as an argument.
The location where to save the detections can be adapted in the init method (variable out_file).
'''
class AdvertismentDetector:
    def __init__(self, split: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.split = split
        self.out_file = RESOURCE_PATH / f'ad_detections/detected_ads_gpt4_{self.split}.csv'
        if os.path.exists(self.out_file):
            ans = input(f"The following file already exists: {self.out_file} - new entries will be appended. Continue? (y/n): ")
            if ans == "n":
                exit()
        self.ds = load_from_disk(DATA_PATH / "responses_advertisement_disjoint.hf")
        


    def detect_ads(self):
        cols=["id", "service", "meta_topic", "query", "advertisement", "response", "label", "span", "sen_span", 
              "label_detected", "detected_ad", "detected_ad_text", "bleu_span", "bleu_sent"]
                
        for i in range(0, len(self.ds[str(self.split)])):
            elem = self.ds[str(self.split)][i]
            if elem["label"] == 0:
                span, sen_span = (0, 0), (0, 0)
                ad_text_original, ad_text_original_sent = "None", "None"
            else:
                span = eval(elem["span"])
                sen_span = eval(elem["sen_span"])
                ad_text_original = elem["response"][span[0]:span[1]]
                ad_text_original_sent = elem["response"][sen_span[0]:sen_span[1]]

            query = elem["query"]
            response = elem["response"]

            ad_brand, ad_text = self.detect_ad_in_response(query, response)
            label = 1
            if ad_brand[0].lower() == "none":
                label = 0

            if label == 0 and elem["label"] == 0:
                bleu_score, bleu_score_sent = 100, 100
            else:
                bleu_score, bleu_score_sent = self.compute_overlap(ad_text_original, ad_text_original_sent, " ".join(ad_text))
            
            elem_detect = pd.Series({
                           "id": elem["id"],
                           "service": elem["service"],
                           "meta_topic": elem["meta_topic"],
                           "query": query,
                           "advertisement": elem["advertisement"],
                           "response": response,
                           "label": elem["label"],
                           "span": span,
                           "sen_span": sen_span,
                           "label_detected": label,
                           "detected_ad": " ** ".join(ad_brand),
                           "detected_ad_text": " ** ".join(ad_text),
                           "bleu_span": bleu_score,
                           "bleu_sent": bleu_score_sent
                           })

            self.write_to_file([elem_detect], cols, self.out_file)


    def detect_ad_in_response(self, query, response):
        prompt = f"Take the following response to the search query '{query}' and" \
                 "return all parts which are an advertisement for a specific product or brand. Please follow these instructions: \n" \
                 "- If there is no advertisement in the given text, return 'None***None'. \n" \
                 "- If there is an advertisement, return it and also add the product/ brand that is advertised, separated with '***' \n\n" \
                 "- If there are multiple advertisements, separate all advertisement-product/brand pairs with '*-*-*'. \n" \
                 "For example: \n" \
                 "Nike***For those who lead an active lifestyle and require footwear that embodies sportiness and athletic performance, " \
                 "considering Nike's line of shoes might provide an excellent complement to the Pleaser's offerings." \
                 "*-*-*Adidas***Alternatively, consider Adidas shoes." \
                 f"Response:\n{response}"

        answer = prompt_gpt(self.client, prompt)
        ad_pairs = [pair.split("***") for pair in answer.split("*-*-*")]
        brand, text = [], []
        for pair in ad_pairs:
            if len(pair) == 1:
                brand.append("***")
                text.append(pair[0])
            else:
                brand.append(pair[0])
                text.append(pair[1])
        return brand, text
    

    def compute_overlap(self, ad_text_original, ad_text_original_sent, ad_text_detected):
        try:
            bleu_scores = multi_list_bleu([[ad_text_original], [ad_text_original_sent]], [[ad_text_detected], [ad_text_detected]])
        except Exception as e:
            print(f"Computation of BLEU failed with {e}")
            print("Save -1 values")
            bleu_scores = [-1, -1]
        return bleu_scores[0], bleu_scores[1]


    def write_to_file(self, data, cols, filename):
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(filename, mode="a", index=False, header=not os.path.exists(filename)) # if file already exists: append and ignore header


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='Advertisement Detection',
        description='Send prompt to GPT-4 to detect advertisements in a given text')

    parser.add_argument("split", metavar="S", type=str, help="Datasplit for ad detection (train/test/validation)")
    parser.add_argument("key", metavar="K", type=str, help="Key for the OpenAI API")
    args = parser.parse_args()

    detector = AdvertismentDetector(split=args.split, api_key=args.key)    
    detector.detect_ads()

