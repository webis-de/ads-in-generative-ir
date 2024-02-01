from datasets import Dataset, DatasetDict
import networkx as nx
import numpy as np
import os
import pandas as pd
from typing import Dict

from ads_in_generative_ir import PROJECT_PATH, RESOURCE_PATH
DATA_PATH = PROJECT_PATH / 'data'
META_TOPICS = ["banking", "car", "gaming", "healthcare", "real_estate", "restaurant", "shopping", "streaming",
               "vacation", "workout"]

class ResponseDataset:
    def __init__(self, disjoint_col: str = "advertisement", minimize_col: str = "query", train_size: float = 0.7):
        """
        :param disjoint_col:    Column for which the values will end up in different splits
        :param minimize_col:    Column for which overlap between splits should be minimized (exclusivity not guaranteed)
        :param train_size:      Share of responses to be assigned to the train split (default: 0.7);
                                Test and validation split is 50/50 on the remaining responses
        """
        assert disjoint_col != minimize_col, print("Please provide different values for disjoint_col and minimize_col.")
        self.disjoint_col = disjoint_col
        self.minimize_col = minimize_col
        self.train_size = train_size
        self.prepare_dfs()

    def prepare_dfs(self):
        # Record values in the disjoint column that are already assigned to a split (for cross meta topic consistency)
        assigned_values = {}

        # Load all available Dataframes (original and ad responses) into a single dataframe
        df = pd.DataFrame()
        print("Preparing response splits...")
        for meta_topic in META_TOPICS:
            try:
                t_df = self.load_meta_topic_df(meta_topic=meta_topic, assigned_values=assigned_values)
                df = pd.concat([df, t_df], ignore_index=True, axis=0)
                print(f"- {meta_topic} responses complete")

                # Update the assigned_values in the disjoint column that are already assigned to a split
                assigned_values.update({
                    x[0]: x[1] for x in
                    t_df.loc[t_df[self.disjoint_col].notna(), [self.disjoint_col, "split"]].values.tolist()
                })

            except Exception as e:
                print(f"Failed loading responses for {meta_topic} with Exception: {e}")

        # Split the Dataframe
        self.train_df = df.loc[df["split"] == "train"].drop(["split"], axis=1)
        self.val_df = df.loc[df["split"] == "val"].drop(["split"], axis=1)
        self.test_df = df.loc[df["split"] == "test"].drop(["split"], axis=1)

        # Remove any remaining rows with overlap on the disjoint column
        train_x = self.train_df[self.disjoint_col].unique().tolist()
        val_x = self.val_df[self.disjoint_col].unique().tolist()
        self.val_df = self.val_df.loc[(~self.val_df[self.disjoint_col].isin(train_x))
                                      | (self.val_df[self.disjoint_col].isna())]
        self.test_df = self.test_df.loc[~self.test_df[self.disjoint_col].isin([*train_x, *val_x])|
                                        (self.test_df[self.disjoint_col].isna())]

    def load_meta_topic_df(self, meta_topic: str, assigned_values: Dict[str, str] = None, iterations: int = 5):
        # Create multiple versions of the ads dataframe and pick the one with the lowest overlap on the minimize column
        min_overlap = float('inf')
        df_a = pd.DataFrame()

        for i in range(iterations):
            tmp_df_a = split_ad_responses(meta_topic=meta_topic,
                                          disjoint_col=self.disjoint_col,
                                          minimize_col=self.minimize_col,
                                          train_size=self.train_size,
                                          assigned_values=assigned_values)
            overlap = self._count_minimize_col_overlap(tmp_df_a)
            if overlap < min_overlap:
                df_a = tmp_df_a
                min_overlap = overlap

        # Split the original responses based on the split of the ads responses
        df_o = split_or_responses(df_a=df_a, meta_topic=meta_topic, train_size=self.train_size)
        return pd.concat([df_a, df_o], ignore_index=True)


    def _count_minimize_col_overlap(self, df: pd.DataFrame):
        train = df.loc[df["split"] == "train", self.minimize_col].unique().tolist()
        val = df.loc[df["split"] == "val", self.minimize_col].unique().tolist()
        test = df.loc[df["split"] == "test", self.minimize_col].unique().tolist()

        return len([x for x in train if x in val]) + \
               len([x for x in val if x in test]) + \
               len([x for x in train if x in test])

    def _create_ds(self):
        if not hasattr(self, "train_df"):
            self.prepare_dfs()
        # Turn individual DataFrames into datasets join the datasets into a DatasetDict
        self.dataset = DatasetDict(
            {
                'train': Dataset.from_pandas(self.train_df, preserve_index=False),
                'test': Dataset.from_pandas(self.test_df, preserve_index=False),
                'validation': Dataset.from_pandas(self.val_df, preserve_index=False)
            }
        )
        print("\nCreated DatasetDict of responses")


    def save_dataset(self):
        if not hasattr(self, "dataset"):
            self._create_ds()

        out_path = DATA_PATH / f"responses_{self.disjoint_col}_disjoint.hf"
        self.dataset.save_to_disk(out_path)
        print(f"\nSaved dataset to {out_path}")

def split_ad_responses(meta_topic: str, disjoint_col: str = "advertisement", minimize_col: str = "query",
                       train_size: float = 0.7, assigned_values: Dict[str, str] = None):
    """
    Prepare the DataFrame of ads responses for a meta topic and split them into train, validation, and test.
    :param meta_topic:      Meta topic for which to split responses
    :param disjoint_col:    Column for which the values will end up in different splits
    :param minimize_col:    Column for which overlap between splits should be minimized (exclusivity not guaranteed)
    :param train_size:      Share of responses to be assigned to the train split (default: 0.7);
                            Test and validation split is 50/50 on the remaining responses
    :param assigned_values: Optional dictionary that restricts selected values to a specific split
                            Used to ensure consistent splitting across meta topics
    :return:                A dataframe with splits
    """
    # 1. Load, filter, and prepare the dataframe
    advertisement_df = pd.read_csv(RESOURCE_PATH / f"responses_with_ads/{meta_topic}_injected.csv")
    df_a = advertisement_df.rename(columns={"ad_response": "response", "response_id": "id"})
    df_a = df_a.loc[(df_a["span"] != '(0, 0)') & (df_a["num_inj_sen"] == 1)]
    df_a["label"] = 1
    df_a["meta_topic"] = meta_topic
    df_a = df_a[["id", "service", "meta_topic", "query", "advertisement", "response", "label", "span", "sen_span"]]

    # 2. Keep only one ad response per original response
    df_a1 = df_a.loc[df_a.groupby("id")["query"].transform("count") > 1].groupby("id").first().reset_index()
    df_a2 = df_a.loc[df_a.groupby("id")["query"].transform("count") == 1]
    df_a = pd.concat([df_a1, df_a2], ignore_index=True)

    # 3. Create a joined dataframe to group by the disjoint column and collect group-specific values of the minimize column
    df_j = df_a.loc[:, ["id", disjoint_col]].merge(df_a.loc[:, ["id", minimize_col]], on="id")
    df_j = df_j.groupby(disjoint_col)[minimize_col].agg(list).reset_index()

    # 4. Count the number of shared elements in the minimize column for each value in the disjoint column
    # The number of shared elements will be weight of the edge between two values in the disjoint column
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    minimize_tuples = []
    for _, row1 in df_j.iterrows():
        for _, row2 in df_j.iterrows():
            num_shared_elem = len(intersection(row1[minimize_col], row2[minimize_col]))
            minimize_tuples.append((row1[disjoint_col], row2[disjoint_col], num_shared_elem))

    # 5. Build a graph from the minimize tuples and find communities of values in the disjoint_col as basis for splitting
    G = nx.Graph()
    for disjoint_val_a, disjoint_val_b, num_shared_elem in minimize_tuples:
        if disjoint_val_a != disjoint_val_b:
            G.add_edge(disjoint_val_a, disjoint_val_b, weight=num_shared_elem)
    communities = [list(com) for com in nx.community.louvain_communities(G)]

    # 6. Map the rows of the dataframe to their community value and based on that to a split
    community_mapping = {val: i for i, c in enumerate(communities) for val in c}
    df_a["community"] = df_a[disjoint_col].apply(lambda val: community_mapping[val])
    counts_per_community = df_a.groupby("community")["id"].count().reset_index().sort_values("id",
                                                                                           ascending=False).values.tolist()
    # Optional: Pre-assign communities to a certain split based on the assigned values
    assigned_communities = {}
    if assigned_values:
        for val, split in assigned_values.items():
            val_df = df_a.loc[df_a[disjoint_col] == val, "community"]
            if len(val_df) > 0:
                assigned_communities[df_a.loc[df_a[disjoint_col] == val, "community"].iloc[0]] = split

    split_mapping = _distribute_communities(counts_per_community, train_size=train_size,
                                            assigned_communities=assigned_communities)
    df_a["split"] = df_a["community"].apply(lambda c: split_mapping[c])

    return df_a.drop("community", axis=1)

def _distribute_communities(counts_per_community, train_size: float = 0.7, assigned_communities: Dict[int, str] = None):
    """
    Distribute communities of ads into train, validation, and test
    :param counts_per_community:    List of pairs (community, number of responses in community)
    :param train_size:              Share of responses to be assigned to the train split (default: 0.7);
                                    Test and validation split is 50/50 on the remaining responses
    :param assigned_communities:    Optional dictionary that restricts selected communities to a specific split
                                    Used to ensure consistent splitting across meta topics
    :return:                        Mapping from community to split
    """
    test_size = val_size = (1 - train_size) / 2
    desired_split = np.array([train_size, val_size, test_size])
    communities = [[], [], []]
    counts = np.array([0, 0, 0])

    labels = ["train", "val", "test"]
    # Optional: Distribute assigned communities up front
    if assigned_communities:
        for community, split in assigned_communities.items():
            assigned_split = labels.index(split)
            elem = [x for x in counts_per_community if x[0] == community][0]
            communities[assigned_split].append(elem[0])
            counts[assigned_split] += elem[1]
            counts_per_community.remove(elem)

    # Distribute the communities to splits
    for community, count in counts_per_community:
        actual_split = counts / max(1, np.sum(counts))
        target_split = np.argmin(actual_split - desired_split)

        # Assign the community to the selected split and update counts
        communities[target_split].append(community)
        counts[target_split] += count


    return {c: labels[i] for i, split_cs in enumerate(communities) for c in split_cs}


def split_or_responses(df_a: pd.DataFrame, meta_topic: str = "banking", train_size: float = 0.7):
    """
    Prepare the DataFrame of ads responses for a meta topic and split them into train, validation, and test.
    :param df_a:            DataFrame of ad responses, split into train, validation, and test
    :param meta_topic:      Meta topic for which to split responses
    :param train_size:      Share of responses to be assigned to the train split (default: 0.7);
                            Test and validation split is 50/50 on the remaining responses
    :return:                A dataframe with splits
    """
    # 1. Load and prepare the dataframe
    original_df = pd.read_csv(RESOURCE_PATH / f"generated_responses/{meta_topic}_filtered.csv")
    df_o = original_df.rename(columns={"response_id": "id"})
    df_o["label"] = 0
    df_o["meta_topic"] = meta_topic
    df_o = df_o[["id", "service", "meta_topic", "query", "response", "label"]]

    # 2. Map original responses into the same split as their ad counterparts (using the ids and then the queries)
    # For the queries, apply train last to have any, potentially duplicated queries in the train split
    split_groups = df_a.groupby("split")
    id_mapping = {id_.replace("-A", "-N"): split
                  for split, ids in split_groups["id"].agg(list).reset_index().values
                  for id_ in ids}
    query_mapping = {}
    query_df = split_groups["query"].agg(list).reset_index()
    for split in ["test", "val", "train"]:
        queries = query_df.loc[query_df["split"] == split]["query"].iloc[0]
        for query in queries:
            query_mapping[query] = split

    df_o["split"] = df_o.apply(lambda row: id_mapping.get(row["id"], query_mapping.get(row["query"], None)),
                               axis=1)

    df = pd.concat([df_a, df_o.loc[df_o["split"].notna()]])

    # 3. Distribute the remaining responses to achieve the desired shares
    share_dict = {p: count for p, count in
                  df.groupby("split")["query"].count().reset_index().values.tolist()}
    remaining_df = df_o.loc[df_o["split"].isna()]

    n_remain = remaining_df.shape[0]
    abs_train_size = round(train_size * (n_remain + sum(share_dict.values())) - share_dict["train"])
    test_size = 0.5 * (1 - train_size)
    abs_test_size = round(test_size * (n_remain + sum(share_dict.values())) - share_dict["test"])

    or_tuples = remaining_df[["id", "query"]].values.tolist()
    query_mapping = _distribute_or_tuples(or_tuples, train_size=abs_train_size, test_size=abs_test_size)
    remaining_df.loc[:, "split"] = remaining_df["query"].apply(lambda q: query_mapping[q])

    return pd.concat([df_o.loc[df_o["split"].notna()], remaining_df], ignore_index=True)

def _distribute_or_tuples(response_list, train_size: int, test_size: int):
    """
    Distribute responses into train, validation, and test while ensuring that the responses with the same query
    end up in the same split.
    :param response_list:   List of tuples (ID, Query)
    :param train_size:      Absolute number of responses to be assigned to the train split
    :param test_size:       Absolute number of responses to be assigned to the test split
                            Test and validation split is 50/50 on the remaining responses

    :return:               Mapping from query to split
    """
    num_responses = len(response_list)
    val_size = num_responses - train_size - test_size
    desired_split = np.array([train_size / num_responses, val_size / num_responses, test_size / num_responses])

    counts = np.array([0, 0, 0])
    queries = [[], [], []]
    ids = [[], [], []]

    for id_, query in response_list:
        # Check if the query is already in a split (Can only be true for one q_list as the responses are added iteratively)
        query_splits = [i for i, q_list in enumerate(queries) if query in q_list]
        if query_splits:
            target_split = query_splits[0]
        else:
            actual_split = counts / max(1, np.sum(counts))
            target_split = np.argmin(actual_split - desired_split)

        # Assign the response to the selected split and update counts
        ids[target_split].append(id_)
        counts[target_split] += 1
        queries[target_split].append(query)

    labels = ["train", "val", "test"]
    return {q: labels[i] for i, split_qs in enumerate(queries) for q in split_qs}

if __name__ == "__main__":
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    import argparse
    parser = argparse.ArgumentParser(
        prog='Response Dataset',
        description='Create a dataset of responses')

    parser.add_argument('-d', '--disjoint_col', type=str, default='advertisement',
                        help='Column for which the values will end up in different splits. ' 
                             'Default value is advertisement.')
    parser.add_argument('-m', '--minimize_col', type=str, default='query',
                        help='Column for which overlap between splits should be minimized. '
                             'Needs to be different from disjoint_col.')
    args = parser.parse_args()

    r = ResponseDataset(disjoint_col=args.disjoint_col, minimize_col=args.minimize_col)
    r.save_dataset()