import pandas as pd
from datasets import load_from_disk
from chatnoir_api.chat import ChatNoirChatClient
import os.path

from ads_in_generative_ir import RESOURCE_PATH, PROJECT_PATH
DATA_PATH = PROJECT_PATH / "data"

'''
The ad detection with alpaca is run via the ChatNoir API which can be installed with 'pip3 install chatnoir_api'
(https://webis.de/publications.html#bevendorff_2018).
The location where to save the detections can be adapted in the init method (variable out_file).
'''
class AdvertismentDetector:
    def __init__(self, split: str):
        self.client = ChatNoirChatClient()
        self.split = split
        self.out_file = RESOURCE_PATH / f'ad_detections/detected_ads_alpaca_{self.split}.csv'
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
            query = elem["query"]
            response_unicode = elem["response"]
            response_encode = response_unicode.encode("ascii", "ignore")
            response = response_encode.decode()
            print(i, "id:", elem["id"])

            ad_brand, ad_text = self.detect_ad_in_response(response.replace("Is there anything else I can help you with?", ""))

            label = 1
            bleu_score, bleu_score_sent = 0, 0

            elem_detect = pd.Series({
                           "id": elem["id"],
                           "service": elem["service"],
                           "meta_topic": elem["meta_topic"],
                           "query": query,
                           "advertisement": elem["advertisement"],
                           "response": response,
                           "label": elem["label"],
                           "span": elem["span"],
                           "sen_span": elem["sen_span"],
                           "label_detected": label,
                           "detected_ad": ad_brand, #" ** ".join(ad_brand),
                           "detected_ad_text": ad_text, #" ** ".join(ad_text),
                           "bleu_span": bleu_score,
                           "bleu_sent": bleu_score_sent
                           })

            self.write_to_file([elem_detect], cols, self.out_file)
    

    def detect_ad_in_response(self, response):
        prompt = f"Check if the following text contains any advertisements. If yes, return the advertised product, otherwise return None: {response}"
        answer = self.client.chat(prompt)
        return "X", answer


    def write_to_file(self, data, cols, filename):
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(filename, mode="a", index=False, header=not os.path.exists(filename)) # if file already exists: append and ignore header


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='Advertisement Detection',
        description='Send prompt to Alpaca to detect advertisements in a given text')

    parser.add_argument("split", metavar="S", type=str, help="Datasplit for ad detection (train/test/validation)")
    args = parser.parse_args()

    detector = AdvertismentDetector(split=args.split)    
    detector.detect_ads()