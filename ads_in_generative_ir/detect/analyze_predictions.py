import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score
import math

from ads_in_generative_ir import RESOURCE_PATH
AD_PATH = RESOURCE_PATH / "/ad_detections"


'''
To evaluate the effectiveness the LLM detections, precision, recall and other scores can be computed.
Make sure to define the correct input and output paths in the init method (file, file_bleu and suffixes if necessary).
'''
class Evaluator:
    def __init__(self, split, model, eval, save, count):
        self.split = split
        if eval == "topics":
            self.split = "all"
        self.model = model
        self.eval = eval
        self.save = save
        self.count = count
        if self.model == "gpt4":
            self.suffix = "_v2"
        elif self.model == "mistral":
            self.suffix = "_2_postprocessed"
        else:
            self.suffix = "_postprocessed"

        self.file = AD_PATH / f'{model}/detected_ads_{model}_{split}{self.suffix}.csv'
        self.file_bleu = AD_PATH / f'{model}/detected_ads_{model}_{split}{self.suffix}_corrected_bleu.csv'
        self.df = pd.read_csv(self.file)


    def evaluate(self):
        # print(f"number of bing queries in {self.split} =", list(self.df["service"]).count("bing"), \
        #       "- youchat queries:", list(self.df["service"]).count("youchat"))

        if self.eval == "bleu":
            self.recompute_bleu_scores(pd.read_csv(self.file), self.file_bleu)
            # self.get_average_bleu_score(pd.read_csv(self.file_bleu))

        elif self.eval == "test":
            self.evaluate_one_split(self.df)
            
        elif self.eval == "topics" or self.eval == "all":
            path_pre = str(AD_PATH / f'{self.model}/detected_ads_{self.model}_')
            df_train = pd.read_csv(path_pre + f'train{self.suffix}.csv') 
            df_dev = pd.read_csv(path_pre + f'validation{self.suffix}.csv') 
            df_test = pd.read_csv(path_pre + f'test{self.suffix}.csv')
            df_merged = pd.concat([df_train, df_dev, df_test])
            if self.eval == "topics":
                self.evaluate_per_topic(df_merged)
            else:
                self.evaluate_one_split(df_merged)

        elif self.eval == "vote":
            self.get_majority_voting()
        else:
            print("unknown evaluation scenario")


    # separate df into topics and compute scores per topic
    def evaluate_per_topic(self, df):
        data = []
        topics = ["banking", "car", "gaming", "healthcare", "real_estate", "restaurant", "shopping", "streaming", "vacation", "workout"]
        for topic in topics:
            print("\nTOPIC:", topic)
            df_topic = df.loc[df['meta_topic'] == topic]
            print("(len:", len(df_topic["id"]), "elements)")
            prec, recall = self.evaluate_one_split(df_topic)
            data.append([topic, prec, recall])
        df_new = pd.DataFrame(data, columns=["topic", "precision", "recall"])
        df_new.to_csv("./scores_" + self.model + ".csv")

    def evaluate_one_split(self, df):
        cm = self.get_confusion_matrix(df, {"TP": 0, "TN": 0, "FP": 0, "FN": 0})
        prec, recall = self.get_scores(df, cm)
        if self.count == "True":
            self.count_preds(df)
        return prec, recall

    def get_confusion_matrix(self, df, cm):
        for i, row in df.iterrows():
            label = row["label"]
            pred = row["label_detected"]
            if label == 1 and pred == 1:
                cm["TP"] += 1
            elif label == 1 and pred == 0:
                cm["FN"] += 1
            elif label == 0 and pred == 1:
                cm["FP"] += 1
            elif label == 0 and pred == 0:
                cm["TN"] += 1
            else:
                print("Something is wrong with labels in row", i)
        print("Confusion matrix:", cm)
        return cm

    def get_scores(self, df, cm):
        true_labels = list(df["label"])
        pred_labels = list(df["label_detected"])
        f1_ma = f1_score(true_labels, pred_labels, average="macro")
        f1_mi = f1_score(true_labels, pred_labels, average="micro")
        # f1_w = f1_score(true_labels, pred_labels, average="weighted")
        precision = cm["TP"] / ( cm["TP"] + cm["FP"] )
        recall = cm["TP"] / ( cm["TP"] + cm["FN"] )
        print("F1 macro:", round(f1_ma, 5), "- F1 micro:", round(f1_mi, 5))
        # print("F1 weighted:", round(f1_w, 5))
        print("\nPrecision:", round(precision, 5), "- Recall:", round(recall, 5), "\n")
        if self.model == "gpt4":
            bleu, bleu_sub = self.get_average_bleu_score(df)
            print("Avg. Bleu:", round(bleu, 4))
            print("Avg. Bleu on correct predicitons:", round(bleu_sub, 4))
        return precision, recall

    # iterate dataframe and recompute bleu scores
    def recompute_bleu_scores(self, df, file_new):
        for i, row in df.iterrows():
            span = eval(row["span"])
            sen_span = eval(row["sen_span"])
            if isinstance(row["detected_ad_text"], str):
                bleus, bleu = self.compute_bleu(row["response"][span[0]:span[1]], row["response"][sen_span[0]:sen_span[1]], row["detected_ad_text"])
                df.at[i, "bleu_sent"] = str([round(b, 3) for b in bleus])
                df.at[i, "bleu_span"] = round(bleu, 3)
            else:
                df.at[i, "bleu_sent"] = [round(float(row["bleu_span"]) / 100, 3)]
                df.at[i, "bleu_span"] = round(float(row["bleu_span"]) / 100, 3)

        df.to_csv(file_new, index=False)

    def compute_bleu(self, ad_text_original, ad_text_original_sent, ad_text_detected):
        detected = [word_tokenize(text) for text in ad_text_detected.split("**")]
        scores = []
        try:
            for det in detected:
                scores.append(sentence_bleu([word_tokenize(ad_text_original), word_tokenize(ad_text_original_sent)], det))
        except Exception as e:
            print(f"Computation of BLEU sentence failed with {e}")
            print("Save -1 values")
            scores = [-1]
        try:
            score = sentence_bleu([word_tokenize(ad_text_original), word_tokenize(ad_text_original_sent)], word_tokenize(ad_text_detected))
        except Exception as e:
            print(f"Computation of BLEU corpus failed with {e}")
            print("Save -1 values")
            score = -1
        return scores, score

    def get_average_bleu_score(self, df):
        bleus = [float(b.replace(",", ".")) for b in list(df["bleu_span"])]
        # bleus = list(df["bleu_span"])
        avg = sum(bleus) / len(bleus)
        # print("avg bleu", split, round(avg,4))
        # BLEU only on correct ad predictions:
        df_sub = df.loc[(df["label"] == 1) & (df["label_detected"] == 1)]
        bleus_sub = [float(b.replace(",", ".")) for b in list(df_sub["bleu_span"])]
        # bleus_sub = list(df:sub["bleu_span"])
        avg_sub = sum(bleus_sub) / len(bleus_sub)
        return avg, avg_sub

    def count_preds(self, df):
        path_pre = str(AD_PATH / f'{self.model}/subsets/detected_ads_{self.model}_{self.split}_')
        self.count_additional_ads(df, path_pre + f'additional_ads.csv')
        self.count_missing_ads(df, path_pre + f'missing_ads.csv')
        self.count_multiple_ads(df, path_pre + f'multiple_ads.csv')

    # count and save examples where gpt4 detects multiple ads
    def count_multiple_ads(self, df, outfile):
        data = []
        for i, row in df.iterrows():
            if "**" in str(row["detected_ad"]):
                data.append(row)
        df_sub = pd.DataFrame(data, columns=list(df))
        print("multiple ads:", len(list(df_sub["id"])), "(", \
                round(len(list(df_sub["id"]))/ len(list(df["id"]))*100, 1) ,"%)", \
                "-> bing:", list(df_sub["service"]).count("bing"), \
                "youchat:", list(df_sub["service"]).count("youchat"))
        if self.save == "True":
            df_sub.to_csv(outfile, index=False)

    # count and save examples where no ad should be present, but gpt4 detects one
    def count_additional_ads(self, df, outfile):
        df_sub = df.loc[(df["label"] == 0) & (df["label_detected"] == 1)]
        print("additional ads:", len(list(df_sub["id"])), "(", \
                round(len(list(df_sub["id"]))/ len(list(df["id"]))*100, 1) ,"%)" \
                "-> bing:", list(df_sub["service"]).count("bing"), \
                "youchat:", list(df_sub["service"]).count("youchat"))
        if self.save == "True":
            df_sub.to_csv(outfile)
        # count_query_ad_overlaps(df_sub)

    # count and save examples with ads that gpt4 did not detect
    def count_missing_ads(self, df, outfile):
        df_sub = df.loc[((df["label"] == 1) & (df["label_detected"] == 0))]
        print("missing ads:", len(list(df_sub["id"])), "(", \
                round(len(list(df_sub["id"]))/ len(list(df["id"]))*100, 1) ,"%)" \
                "-> bing:", list(df_sub["service"]).count("bing"), \
                "youchat:", list(df_sub["service"]).count("youchat"))
        if self.save == "True":
            df_sub.to_csv(outfile)


    # count and save examples where brand/product requested in query is considered as ad
    def count_query_ad_overlaps(self, df):
        count_pure, count_sub = 0, 0
        for i, row in df.iterrows():
            query = str(row["query"])
            ad_pred = str(row["detected_ad"])
            if query.lower() == ad_pred.lower():
                count_pure += 1
            elif query.lower() in ad_pred.lower() or ad_pred.lower() in query.lower():
                count_sub += 1
        print("pure overlaps:", count_pure)
        print("partial overlaps:", count_sub)


    # combine predictions of three LLMs
    def get_majority_voting(self):
        dfg = pd.read_csv(AD_PATH / f'gpt4/detected_ads_gpt4_test_v2.csv')
        dfa = pd.read_csv(AD_PATH / f'alpaca/detected_ads_alpaca_test.csv')
        dfm = pd.read_csv(AD_PATH / f'mistral/detected_ads_mistral_test_2.csv')
        labelsg = list(dfg.sort_values("id")["label_detected"])
        labelsa = list(dfa.sort_values("id")["label_detected"])
        labelsm = list(dfm.sort_values("id")["label_detected"])
        labels_gt = list(dfg.sort_values("id")["label"])
        ensemble_pred = []
        for i, triple in enumerate(zip(labelsg, labelsa, labelsm)):
            if math.isnan(triple[1]):
                print(triple)
                continue
            triple = [int(t) for t in triple]
            pred = 0
            if triple.count(1) > 1:
            # if triple.count(1) == 3:
                pred = 1
            ensemble_pred.append(pred)
            
        intersec = [1 if (pair[0] == 1 and pair[1] == 1) else 0 for pair in zip(labels_gt, ensemble_pred)]
        prec = intersec.count(1) / ensemble_pred.count(1)
        rec = intersec.count(1) / labels_gt.count(1)
        print("precision =", prec)
        print("recall =", rec)
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("split", metavar="S", type=str, help="Datasplit (train/test/validation)")
    parser.add_argument("model", metavar="M", type=str, help="Model (gpt4/alpaca/mistral)")
    parser.add_argument("eval", metavar="E", type=str, help="what to evaluate (bleu/test/topics/all/vote)")
    parser.add_argument("save", metavar="save", type=str, default=False, help="save FP and FN responses")
    parser.add_argument("count", metavar="C", type=str, default=False, help="count missing/additional/multiple predictions")
    args = parser.parse_args()

    evaluator = Evaluator(split=args.split, model=args.model, eval=args.eval, save=args.save, count=args.count)   
    evaluator.evaluate()

