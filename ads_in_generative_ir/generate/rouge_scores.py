from datasets import load_from_disk
from itertools import combinations
import pandas as pd
import spacy
from tqdm import tqdm

from ads_in_generative_ir import PROJECT_PATH

DATA_PATH = PROJECT_PATH / "data"
nlp = spacy.load("en_core_web_sm")

# ========================================
# Data preparation
# ========================================
def load_df():
    """
    Load the dataset of responses and get a DataFrame with all responses that contain an ad
    """
    ds = load_from_disk(DATA_PATH / "responses_advertisement_disjoint.hf")
    df = pd.DataFrame()
    for split in ["train", "validation", "test"]:
        s_df = pd.DataFrame(ds[split])
        s_df["split"] = split
        df = pd.concat([df, s_df], ignore_index=True)

    return df.loc[df["label"] == 1]

def get_injection(row):
    """
    Get the injection part of a response
    """
    sen_span = eval(row["sen_span"])
    return row["response"][sen_span[0]:sen_span[1]]

def get_cleaned_injection(row):
    """
    Get the ad injection and remove the name of the brand/product
    """
    injection = get_injection(row)
    return injection.replace(row["advertisement"], "")

def get_lemmas(row):
    """
    Lemmatize the injection and return the result as a list
    """
    injection = row["injection"]
    lemmas = [token.lemma_.lower() for token in nlp(injection) if not (token.is_stop or token.is_punct)]
    if " " in lemmas:
        lemmas.remove(" ")
    return lemmas

def lemmatize_responses(df):
    tqdm.pandas()
    df["injection"] = df.progress_apply(lambda row: get_cleaned_injection(row), axis=1)
    df["lemmas"] = df.progress_apply(lambda row: get_lemmas(row), axis=1)
    return df


# ========================================
# ROGUE Score
# ========================================
def rouge1_f1(lemmas1, lemmas2):
    """
    Take two lists of lemmas and compute the ROUGE-1 F1 score
    """
    n_match = len([l for l in lemmas1 if l in lemmas2])
    n_reference = len(lemmas1)
    n_candidate = len(lemmas2)

    r = n_match / n_reference
    p = n_match / n_candidate
    if p == 0 and r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def pairwise_rouge_scores(out_path = DATA_PATH / "rouge_scores.csv"):
    # Load and prepare the data
    df = load_df()
    df = lemmatize_responses(df)

    # Write the CSV Header
    with open(out_path, "w") as f:
        f.write("id1,id2,rouge1-f1\n")

    # Iterate over all pairs of ids
    ids = df["id"].unique()
    total = len(ids) * (len(ids)-1) / 2

    with open(out_path, "a") as f:
        for c in tqdm(combinations(ids, r=2), desc="Calculating ROUGE scores", total=total):
            id1, id2 = c
            # Calculate the score
            lemmas1 = df.loc[df["id"] == id1].iloc[0]["lemmas"]
            lemmas2 = df.loc[df["id"] == id2].iloc[0]["lemmas"]
            f1 = rouge1_f1(lemmas1, lemmas2)

            # Update results
            f.write(f"{id1},{id2},{f1}\n")

def aggregate_rouge_scores():
    df = load_df()
    r_df = pd.read_csv(DATA_PATH / "rouge_scores.csv")

    # Add meta topics to the responses
    r_df = (r_df.merge(df[["id", "meta_topic"]], left_on="id1", right_on="id")
            .rename(columns={"meta_topic": "meta_topic1"})
            .drop("id", axis=1))
    r_df = (r_df.merge(df[["id", "meta_topic"]], left_on="id2", right_on="id")
            .rename(columns={"meta_topic": "meta_topic2"})
            .drop("id", axis=1))

    # Calculate average ROUGE-score between two meta topics
    grouped = r_df.groupby(["meta_topic1", "meta_topic2"])["rouge1-f1"].mean().reset_index()
    grouped.to_csv(DATA_PATH / "rouge_scores_topics.csv")
    print(f"Avg score within the same meta topic: "
          f"{grouped.loc[(grouped['meta_topic1'] == grouped['meta_topic2'])]['rouge1-f1'].mean()}")

    different_topic = grouped.loc[(grouped['meta_topic1'] != grouped['meta_topic2'])]
    print(f"Avg score for different meta topics: "
          f"{different_topic['rouge1-f1'].mean()}")
    max_row = different_topic.sort_values("rouge1-f1", ascending=False).iloc[0]
    print(f"Max score for different topics: "
          f"{max_row['meta_topic1']}, {max_row['meta_topic2']}, {max_row['rouge1-f1']}")

    # Find the pairs of injections with the highest overlap
    df["injection"] = df.apply(lambda row: get_injection(row), axis=1)
    idx = r_df.groupby(['meta_topic1', 'meta_topic2'])['rouge1-f1'].transform("max") == r_df['rouge1-f1']
    max_sim = r_df[idx]
    max_sim = (max_sim.merge(df[["id", "injection"]], left_on="id1", right_on="id")
               .rename(columns={"injection": "injection1"})
               .drop("id", axis=1))
    max_sim = (max_sim.merge(df[["id", "injection"]], left_on="id2", right_on="id")
               .rename(columns={"injection": "injection2"})
               .drop("id", axis=1))
    max_sim.to_csv(DATA_PATH / "rouge_scores_max.csv")


if __name__ == "__main__":
    pairwise_rouge_scores()