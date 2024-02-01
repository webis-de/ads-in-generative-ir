import argparse
from datasets import Dataset, load_from_disk
from itertools import groupby
from operator import itemgetter
import os
import pandas as pd
import spacy
import torch as T
from torch.utils.data import DataLoader

from ads_in_generative_ir import PROJECT_PATH
import ads_in_generative_ir.classify.models as m
from ads_in_generative_ir.classify.model_wrapper import ModelWrapper

# ================================================================
# Preparation
# ================================================================
DATA_PATH = PROJECT_PATH / 'data'
EVALUATION_PATH = PROJECT_PATH / "models/evaluation"
device = T.device("cuda" if T.cuda.is_available() else "cpu")
class EvaluationWrapper(ModelWrapper):
    def __init__(self,  sbert_model: str, disjoint_col: str = "advertisement", holdout_topic: str = None,
                 model_suffix: str = None):
        """
        Wrapper to manage model evaluatuion
        :param sbert_model:         Name to identify a sentence transformer model on Huggingface
                                    (used for model architecture and tokenizer)
        :param disjoint_col:        Column that is disjoint across train, test, and validation split.
                                    Determines which dataset will be used for evaluation
        :param holdout_topic:       Optional: Name of a meta topic to use as the evaluation set.
                                    If none is provided, the default test set will be used for evaluation.
        :param model_suffix:        Optional: Suffix for the model_name. The base name is constructed from other input
        """
        super().__init__(sbert_model=sbert_model, disjoint_col=disjoint_col, holdout_topic=holdout_topic,
                         model_suffix=model_suffix, dataset_prefix="responses")

        # Load the model as baseline from huggingface or weights from a contrastive pre-trained model.
        print("Preparing the model...")
        self.model = m.SupervisedModel(sbert_model=sbert_model)
        self.loss = T.nn.BCEWithLogitsLoss()

        # Define the path to save files
        self.file_prefix = f"{self.disjoint_col}_{sbert_model}"
        if holdout_topic:
            self.file_prefix += f"_{holdout_topic}"
        if model_suffix:
            self.file_prefix += f"_{model_suffix}"
        self._create_evaluation_folders()


    @staticmethod
    def _create_evaluation_folders():
        if not os.path.exists(EVALUATION_PATH):
            os.mkdir(EVALUATION_PATH)
        for folder in ["sentences", "responses"]:
            if not os.path.exists(os.path.join(EVALUATION_PATH, folder)):
                os.mkdir(os.path.join(EVALUATION_PATH, folder))

    def _load_dataset(self):
        """
        Function to load a local HuggingFace dataset. If a holdout_topic was specified, the dataset will be
        redistributed to keep all datapoints for the holdout_topic in the test split.
        Format should be:
        'id', 'service', 'meta_topic', 'query', 'advertisement', 'response', 'label', 'span', 'sen_span'
        """
        # 1. Load and potentially redistribute the dataset of responses
        self.dataset = load_from_disk(DATA_PATH / self.dataset_name)
        if self.holdout_topic:
            self._distribute_ds_by_holdout()

        self.response_ds = self.dataset["test"]

        # 2. Split the responses into sentence pairs
        nlp = spacy.load("en_core_web_sm")
        remove_columns = ["id", "service", "meta_topic", "query", "advertisement", "response", "label", "span",
                          "sen_span"]
        tmp_ds = self.dataset["test"].map(lambda element: self._split_into_sentence_pairs(element=element, nlp=nlp),
                                          remove_columns=remove_columns)
        list_of_dicts = [d for d_list in tmp_ds['sentence_pairs'] for d in d_list]
        self.dataset = (Dataset.from_dict({'pairs': list_of_dicts}).flatten()
                        .rename_column("pairs.sentence1", "sentence1")
                        .rename_column('pairs.sentence2', "sentence2")
                        .rename_column('pairs.label', "label")
                        .rename_column('pairs.id', "id")
                        .rename_column('pairs.service', "service")
                        .rename_column('pairs.meta_topic', "meta_topic")
                        .rename_column('pairs.query', "query")
                        .rename_column('pairs.advertisement', "advertisement")
                        .rename_column('pairs.pair_num', "pair_num"))
        self.df = pd.DataFrame(self.dataset)

    @staticmethod
    def _split_into_sentence_pairs(element, nlp):
        contains_ad = element["label"] == 1
        response = element["response"]
        result = {'sentence_pairs': []}
        ad = element["advertisement"] if element["advertisement"] is not None else "no advertisement"

        base_dict = {"id": element["id"], "service": element["service"], "meta_topic": element["meta_topic"],
                     "query": element["query"], "advertisement": ad}

        if contains_ad:
            try:
                start, end = eval(element["sen_span"])
            except:
                return

            ad_pairs = []

            ad_sen = [str(s).strip() for s in nlp(response[start:end].strip()).sents]
            if start > 0:
                sen_pre = [str(s).strip() for s in nlp(response[:start].strip()).sents]
                result['sentence_pairs'] += [{**base_dict,
                                              "sentence1": sen_pre[i],
                                              "sentence2": sen_pre[i + 1],
                                              "label": 0,
                                              "pair_num": i} for i in
                                             range(len(sen_pre[:-1]))]

                ad_pairs.append({**base_dict,
                                 "sentence1": sen_pre[-1],
                                 "sentence2": ad_sen[0],
                                 "label": 1,
                                 "pair_num": len(sen_pre[:-1])})



            if (end + 2) < len(response.strip()):
                sen_after = [str(s).strip() for s in nlp(response[end:].strip()).sents]
                num_collected_pairs = len(result["sentence_pairs"]) + len(ad_pairs)
                ad_pairs.append({**base_dict,
                                 "sentence1": ad_sen[-1],
                                 "sentence2": sen_after[0],
                                 "label": 1,
                                 "pair_num": num_collected_pairs})

                num_collected_pairs = len(result["sentence_pairs"]) + len(ad_pairs)
                result['sentence_pairs'] += [{**base_dict,
                                              "sentence1": sen_after[i],
                                              "sentence2": sen_after[i + 1],
                                              "label": 0,
                                              "pair_num": i + num_collected_pairs} for i in
                                             range(len(sen_after[:-1]))]


            result['sentence_pairs'] += ad_pairs

        else:
            sentences = [str(s).strip() for s in nlp(response.strip()).sents]
            result['sentence_pairs'] = [{**base_dict,
                                         "sentence1": sentences[i],
                                         "sentence2": sentences[i + 1],
                                         "label": 0,
                                         "pair_num": i}
                                        for i in range(len(sentences[:-1]))]

        return result

    # ----------------------------------------------------------------
    # Dataset preparation
    # ----------------------------------------------------------------
    # Source: https://huggingface.co/docs/transformers/training
    def tokenize_data(self):
        """
        Function to tokenize the dataset and set self.test_ds
        """
        if not hasattr(self, "dataset"):
            self._load_dataset()

        # Set removal columns and apply tokenization
        remove_columns = ["sentence1", "sentence2"]
        tokenized_ds = self.dataset.map(lambda example: self._tokenize_function(example),
                                        remove_columns=remove_columns)
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds = tokenized_ds.with_format("torch")

        # Unsqueeze the label column
        self.test_ds = tokenized_ds.map(lambda example: {"labels": T.unsqueeze(example["labels"], dim=0)},
                                        remove_columns=["labels"])

    def _tokenize_function(self, example):
        # Return the tokenized strings (Note: They are in one array)
        # Pad to the maximum length of the model
        return self.tokenizer(example["sentence1"], example["sentence2"], padding="max_length", truncation=True)

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    def evaluate(self, batch_size: int = 16):
        self.make_predictions(batch_size=batch_size)
        self.create_response_csv()

    def make_predictions(self, batch_size: int = 16):
        if not hasattr(self, "test_ds"):
            self.tokenize_data()

        # Prepare model
        encoder_weights = T.load(self.weight_path)
        model = self.model.to(device)
        model.load_state_dict(encoder_weights)

        # Prepare the two dataloader
        test_dl = DataLoader(self.test_ds.with_format("torch", device=device), batch_size=batch_size)

        # Perform evaluation
        model.eval()
        with T.no_grad():
            # Initialize lists to store batch values
            ids, meta_topics, services, queries, ads, pair_num, labels, predictions = ([] for i in range(8))

            for batch in test_dl:
                # Get batch values
                ids += batch.pop('id')
                meta_topics += batch.pop('meta_topic')
                services += batch.pop('service')
                queries += batch.pop('query')
                ads += batch.pop('advertisement')
                pair_num += [int(num.item()) for num in batch.pop('pair_num')]
                labels += [int(label.item()) for label in batch['labels']]

                # Get the logits from the batch
                predictions += [int(logit.item() > 0.5) for logit in model(batch=batch)]

        self.predictions = pd.DataFrame({"id": ids,
                                         "meta_topic": meta_topics,
                                         "service": services,
                                         "query": queries,
                                         "advertisement": ads,
                                         "pair_num": pair_num,
                                         "label": labels,
                                         "prediction": predictions})


        file_path = EVALUATION_PATH / f"sentences/{self.file_prefix}.csv"
        self.predictions.to_csv(file_path)
        print(f"Saved sentence level predictions to {file_path}")

    def create_response_csv(self):
        if not hasattr(self, "predictions"):
            self.make_predictions()

        # 1. Create a DataFrame of responses with aggregated label, prediction and number of sentence pairs with
        # a positive prediction
        p_df = self.predictions
        id_grouped = p_df.groupby("id")
        response_df = id_grouped[["meta_topic", "service", "query", "advertisement"]].first().reset_index()
        response_df = response_df.merge(id_grouped[["label", "prediction"]].max().reset_index(), on="id")
        response_df = response_df.merge(
            p_df.loc[p_df["prediction"] == 1].groupby("id")["pair_num"].agg(list).reset_index(),
            on="id", how="left"
        )

        # 2. Identify the passage marked as advertisement
        response_df["detected_ad_text"] = response_df.apply(lambda row: self._get_detected_ads(row), axis=1)

        # 3. Save the result
        file_path = EVALUATION_PATH / f"responses/{self.file_prefix}.csv"
        response_df.to_csv(file_path)
        print(f"Saved response level predictions to {file_path}")

    def _get_detected_ads(self, row):
        if type(row["pair_num"]) is float:
            return None
        pair_num = sorted(row["pair_num"])

        # Get all pairs for the id and determine consecutive pair numbers
        id_pairs = self.df.loc[self.df["id"] == row["id"]].sort_values("pair_num")
        consecutive_num = []
        for k, g in groupby(enumerate(pair_num), lambda x: x[0] - x[1]):
            consecutive_num.append(list(map(itemgetter(1), g)))

        # Loop over all lists of consecutive numbers to form sentences
        injections = []
        for num_list in consecutive_num:
            if len(num_list) >= 2:
                injections.append(self._get_detection_multiple(num_list, id_pairs))
            else:
                injections.append(self._get_detection_single(num_list[0], id_pairs))

        return injections

    @staticmethod
    def _get_detection_multiple(num_list, id_pairs):
        injection_rows = id_pairs.loc[id_pairs["pair_num"].isin(num_list)]
        injection_pairs = len(injection_rows)

        sentences = []
        for i, (_, row) in enumerate(injection_rows.iterrows()):
            if i + 1 == injection_pairs:
                sentences.append(row["sentence1"])
            else:
                sentences.append(row["sentence2"])

        return ". ".join(sentences[:-1])

    @staticmethod
    def _get_detection_single(num, id_pairs):
        if num == 0:
            return id_pairs.iloc[0]["sentence1"]
        if num + 1 == len(id_pairs):
            return id_pairs.iloc[-1]["sentence2"]

        row = id_pairs.loc[id_pairs["pair_num"] == num].iloc[0]
        return row["sentence1"] + " " + row["sentence2"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Model evaluation',
        description='Evaluate a sentence transformer on a corpus of sentence pairs')

    parser.add_argument('sbert_model', type=str,
                        help='Name of the sentence transformer architecture. E.g. all-mpnet-base-v2')
    parser.add_argument('-d', '--disjoint_col', type=str, default='advertisement',
                        help='Column for which the values are in different splits. Default value is advertisement.')
    parser.add_argument('-t', '--holdout_topic', type=str, default=None,
                        help='Name of a meta topic to use as the test set. Default is None.')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='Name of the optimizer used in training (adam, sgd). Used to identify the weights.'
                             ' Default adam.')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Number of sentence pairs to process in one batch. Default 16.')
    args = parser.parse_args()

    wrapper = EvaluationWrapper(sbert_model=args.sbert_model,
                                disjoint_col=args.disjoint_col,
                                holdout_topic=args.holdout_topic,
                                model_suffix=args.optimizer)
    wrapper.evaluate(batch_size=args.batch_size)