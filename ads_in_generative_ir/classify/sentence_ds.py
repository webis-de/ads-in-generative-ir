from datasets import Dataset, DatasetDict, load_from_disk
import math
import pandas as pd
import re
import spacy
from tqdm import tqdm

from ads_in_generative_ir import PROJECT_PATH
DATA_PATH = PROJECT_PATH / 'data'

class SentenceDataset:
    def __init__(self, disjoint_col: str = "advertisement"):
        """
        Class that works with the output of the response_ds.py script
        :param disjoint_col:    Column for which the values will end up in different splits.
                                Determines which response dataset will be used as base
        """
        self.disjoint_col = disjoint_col
        self._load_response_ds()
        self.nlp = spacy.load("en_core_web_sm")

    def _load_response_ds(self):
        file_name = DATA_PATH / f"responses_{self.disjoint_col}_disjoint.hf"
        try:
            self.response_ds = load_from_disk(DATA_PATH / file_name)
        except:
            raise FileNotFoundError(f"{DATA_PATH / file_name} not found. "
                                    f"Please run `ads_in_generative_ir.classify.response_ds.py -d={self.disjoint_col}`"
                                    f" first.")

    def create_ds(self, share_ad_responses: float = None, min_len: int = 30):
        """
        Create Huggingface dataset with pairs of sentences from the response dataset.
        :param share_ad_responses:  Optional share that positive samples (sentence pairs with an ad) should
                                    make up in the created dataset. Defaults to the share in the response ds.
        :param min_len:             Minimum number of characters that each sentence has to have.

        The resulting dataset has the following entries:
        - 'id':                 ID of the response the sentences were taken from
        - 'service':            Chat service that generated the response
        - 'meta_topic':         Meta topic of the query
        - 'query':              Query that the response was generated for
        - 'advertisement':      Name of the advertisement in the response or None if no advertisement was injected
        - 'sentence1':          First sentence
        - 'sentence2':          Second sentence
        - 'label':              1 if advertisement was injected; 0 if no advertisement was injected
        """
        pair_datasets = {}

        # Loop over all partitions and create a Dataframe of sentence pairs for each
        for partition in ["train", "test", "validation"]:
            df = pd.DataFrame(self.response_ds[partition])
            ad_df = df.loc[df["label"] == 1]
            or_ids = [i.replace("-A", "-N") for i in ad_df["id"].unique()]
            or_df = df.loc[(~df["id"].isin(or_ids)) & (df["label"] == 0)]

            pair_df = pd.DataFrame()

            # 1. Create sentence pairs for all ad responses
            for _, row in tqdm(ad_df.iterrows(),
                               desc=f"Ad sentence pairs for {partition} partition",
                               total=ad_df.shape[0]):
                pair_df = pd.concat([pair_df, self._process_ad_row(row=row, df=df, min_len=min_len)],
                                    ignore_index=True)

            # 2. Collect additional negative pairs (or_pairs) based on the provided share or the ratio in the df
            if not share_ad_responses:
                num_responses = {label: responses for label, responses in
                                 df.groupby("label")["response"].count().reset_index().values}
                share_ad_responses = num_responses[1] / (num_responses[0] + num_responses[1])

            # Determine how many negative samples (or_pairs) are needed in total and how many are already in the dataset
            num_ad_pairs = pair_df.loc[pair_df["label"] == 1].shape[0]
            num_or_pairs = pair_df.loc[pair_df["label"] == 0].shape[0]
            missing_or_pairs = round(num_ad_pairs / share_ad_responses * (1 - share_ad_responses)) - num_or_pairs

            # Determine number of sentence pairs per available response and set final number of responses
            pairs_per_response = math.ceil(missing_or_pairs / or_df.shape[0])
            num_responses = round(missing_or_pairs / pairs_per_response)

            # Sample the responses and collect pairs for them
            sampled_or_df = or_df.sample(n=num_responses, random_state=0)
            for _, row in tqdm(sampled_or_df.iterrows(),
                               desc=f"Non-Ad sentence pairs for {partition} partition",
                               total=num_responses):
                pair_df = pd.concat([pair_df,
                                     self._process_or_row(row=row, num_pairs=pairs_per_response, min_len=min_len)],
                                    ignore_index=True)

            # 3. Turn the pair_df into a dataset and store it in the dictionary
            pair_datasets[partition] = Dataset.from_pandas(pair_df, preserve_index=False)

        self.dataset = DatasetDict(pair_datasets)
        print("- Created DatasetDict of sentence pairs")


    def _process_ad_row(self, row, df, min_len=30) -> pd.DataFrame:
        """
        Turn an advertisement response into pairs of sentences. Each sentence must be at least min_len chars long.
        """
        # Get generic info
        service = row["service"]
        meta_topic = row["meta_topic"]
        query = row["query"]

        # Get the ad data
        ad_id = row["id"]
        advertisement = row["advertisement"]
        ad_response = row["response"]
        start, end = eval(row["sen_span"])

        # Get the corresponding data about the original response
        or_id = ad_id.replace("-A", "-N")
        or_row = df.loc[df.id == or_id].iloc[0]
        or_response = or_row["response"].strip()
        shared_sub_str = or_response[:start].strip()

        # Construct sentence pairs (Both before and after the insertion)
        result = []
        if start > 0:
            shared_prefix = str([s for s in self.nlp(shared_sub_str).sents][-1]).strip()
            ad_continuation = ad_response[start:end].strip()

            if len(shared_prefix) >= min_len and len(ad_continuation) >= min_len:
                result.append({'id': ad_id, 'service': service, 'meta_topic': meta_topic, 'query': query,
                               'advertisement': advertisement,
                               'sentence1': shared_prefix, 'sentence2': ad_continuation, 'label': 1})

            if end < len(or_response):
                or_continuation = str(next(self.nlp(or_response[start:].strip()).sents)).strip()

                if len(shared_prefix) >= min_len and len(or_continuation) >= min_len:
                    result.append(
                        {'id': or_id, 'service': service, 'meta_topic': meta_topic, 'query': query,
                         'advertisement': None, 'sentence1': shared_prefix, 'sentence2': or_continuation, 'label': 0})

        if (end + 2) < len(ad_response.strip()):
            ad_prefix = str(next(self.nlp(ad_response[start:].strip()).sents)).strip()
            shared_continuation = str(next(self.nlp(ad_response[end:].strip()).sents)).strip()

            continuation_idx = max(or_response.find(shared_continuation[:50]), start)
            or_tmp = or_response[:continuation_idx].strip()

            if len(ad_prefix) >= min_len and len(shared_continuation) >= min_len:
                result.append({'id': ad_id, 'service': service, 'meta_topic': meta_topic, 'query': query,
                               'advertisement': advertisement,
                               'sentence1': ad_prefix, 'sentence2': shared_continuation, 'label': 1})

            try:
                or_prefix = str([s for s in self.nlp(or_tmp).sents][-1]).strip()
            except:
                return pd.DataFrame.from_records(result)

            # Ensure that this original version is not the same as for the shared_prefix above
            or_prefix_dict = {'id': or_id, 'service': service, 'meta_topic': meta_topic, 'query': query,
                              'advertisement': None,
                              'sentence1': or_prefix, 'sentence2': shared_continuation, 'label': 0}
            if or_prefix_dict not in result and len(or_prefix) >= min_len and len(shared_continuation) >= min_len:
                result.append(or_prefix_dict)

        return pd.DataFrame.from_records(result)

    def _process_or_row(self, row, num_pairs: int, min_len=30) -> pd.DataFrame:
        """
        Turn a non-advertisement response into pairs of sentences. Each sentence must be at least min_len chars long.
        """
        # Split the response into sentences and construct as many pairs as defined
        sentences = [str(s).strip() for s in self.nlp(row["response"].strip()).sents]
        result = []
        i = 0

        while len(result) < num_pairs and i < len(sentences)-1:
            sen1 = sentences[i]
            sen2 = sentences[i+1]
            if len(sen1) >= min_len and len(sen2) >= min_len:
                result.append({'id': row["id"],
                               'service': row["service"],
                               'meta_topic': row["meta_topic"],
                               'query': row["query"],
                               'advertisement': None,
                               'sentence1': sen1,
                               'sentence2': sen2,
                               'label': 0})
            i += 1

        return pd.DataFrame.from_records(result)

    def save_dataset(self):
        if not hasattr(self, "dataset"):
            self.create_ds()

        out_path = DATA_PATH / f"sentence_pairs_{self.disjoint_col}_disjoint.hf"
        self.dataset.save_to_disk(out_path)
        print(f"- Saved dataset to {out_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='Sentence Dataset',
        description='Create a dataset of sentence pairs from the dataset of responses')

    parser.add_argument('-d', '--disjoint_col', type=str, default='advertisement',
                        help='Column for which the values are in different splits. Based on response_ds.py. '
                             'Default value is advertisement.')
    parser.add_argument('-a', '--ad_share', type=float, help='Optional share that ad pairs should make up '
                                                             'in the created dataset')
    args = parser.parse_args()

    r = SentenceDataset(disjoint_col=args.disjoint_col)
    r.create_ds(share_ad_responses=args.ad_share)
    r.save_dataset()