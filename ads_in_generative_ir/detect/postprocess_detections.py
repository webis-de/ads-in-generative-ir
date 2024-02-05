import pandas as pd
from ads_in_generative_ir import RESOURCE_PATH


'''
Since the responses of Mistral and Alpaca are not as consistent as the responses of GPT-4, they require a special
postprocessing to extract the relevant information.
Make sure to define the correct input and output paths in the init method (in_file and out_file).
'''
class DetectionPostprocessor:
    def __init__(self, split: str, model: str):
        self.split = split
        self.model = model
        suffix = ""
        if model == "mistral":
            suffix = "_2"
        self.in_file = RESOURCE_PATH / f'ad_detections/{self.model}/detected_ads_{self.model}_{self.split}{suffix}.csv'
        self.out_file = str(self.in_file).replace(".csv", "_postprocessed.csv")


    def postprocess_alpaca(self):
        df = pd.read_csv(self.in_file)
        list_prompt = []
        list_empty = []
        for i, row in df.iterrows():
            if pd.isnull(df.iloc[i,2]):
                list_empty.append(i)
                continue
            ans = str(row["detected_ad_text"]).strip()
            if "If yes, return the advertised product, otherwise return None" in ans:
                list_prompt.append(i)
            if ans == "nan" or "none" in ans.lower() or "text does not contain any advertisements" in ans.lower() \
                    or "No, that's all. Thank" in ans:  # to cover both thank you and thanks
                df.at[i, "label_detected"] = 0
                df.at[i, "detected_ad"] = "None"
                df.at[i, "detected_ad_text"] = "None"
            else:
                df.at[i, "label_detected"] = 1
                df.at[i, "detected_ad"] = "***"
                df.at[i, "detected_ad_text"] = ans

        # print("-> prompt in answer: ", list_prompt)
        # print("-> missing response: ", list_empty)
        df.to_csv(self.out_file, index=False)



    def postprocess_mistral(self):
        df = pd.read_csv(self.in_file)
        list_no_stars = []
        list_multiple_stars = []
        for i, row in df.iterrows():
            ans = row["detected_ad"].strip()
            if ans.startswith("***"):
                ans = ans[3:]
            if ans.endswith("***"):
                ans = ans[:-3]
            if "******" in ans:
                ans = ans.replace("******", "***")

            if "none" in ans.lower():
                df.at[i, "label_detected"] = 0
                df.at[i, "detected_ad"] = "None"
                df.at[i, "detected_ad_text"] = "None"
            elif "*-*-*" in ans:
                ad_pairs = [pair.split("***") for pair in ans.split("*-*-*")]
                for pair in ad_pairs:
                    if len(pair) == 1:
                        df.at[i, "label_detected"] = 1
                        df.at[i, "detected_ad"] = "***"
                        df.at[i, "detected_ad_text"] = pair[0]
                    else:
                        df.at[i, "label_detected"] = 1
                        df.at[i, "detected_ad"] = pair[0]
                        df.at[i, "detected_ad_text"] = pair[1]

            elif ans.count("***") == 1:
                pair = ans.split("***")
                df.at[i, "label_detected"] = 1
                df.at[i, "detected_ad"] = pair[0]
                df.at[i, "detected_ad_text"] = pair[1]
            elif "***" not in ans:
                list_no_stars.append(i)
                df.at[i, "label_detected"] = 1
                df.at[i, "detected_ad"] = "***"
                df.at[i, "detected_ad_text"] = ans
            else:
                df.at[i, "label_detected"] = 1
                df.at[i, "detected_ad"] = "***"
                df.at[i, "detected_ad_text"] = ans
                list_multiple_stars.append(i)

        df.to_csv(self.out_file, index=False)

        print("-> no *** in: ", len(list_no_stars))
        print("-> multiple *** in: ", len(list_multiple_stars))


    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='Advertisement Detection',
        description='Send prompt to GPT-4 to detect advertisements in a given text')

    parser.add_argument("split", metavar="S", type=str, help="Datasplit for ad detection (train/test/validation)")
    parser.add_argument("model", metavar="M", type=str, help="Key for the OpenAI API")
    args = parser.parse_args()

    detector = DetectionPostprocessor(split=args.split, model=args.model)

    if detector.model == "mistral":
        detector.postprocess_mistral()
    elif detector.model == "alpaca":
        detector.postprocess_alpaca()
    else:
        print("unknown model")
        exit()