import difflib
from nltk.tokenize.punkt import PunktSentenceTokenizer
import math
from openai import OpenAI
import pandas as pd
import re
from tqdm import tqdm
from typing import Union

from ads_in_generative_ir import RESOURCE_PATH
from ads_in_generative_ir.generate.utils import prompt_gpt

class AdvertisementInjector:
    def __init__(self, meta_topic: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.meta_topic = meta_topic

        self.out_file = RESOURCE_PATH / f'responses_with_ads/{meta_topic}_injected.csv'

        self.pair_df = pd.read_csv(RESOURCE_PATH / f'advertisements/{meta_topic}_query_pairs.csv')
        self.quality_df = pd.read_csv(RESOURCE_PATH / f'advertisements/{meta_topic}_qualities.csv')
        self.response_df = pd.read_csv(RESOURCE_PATH / f'generated_responses/{meta_topic}_filtered.csv')

    def inject_ads(self):
        # Loop over all unique queries
        queries = self.pair_df["query"].unique().tolist()
        df = pd.DataFrame()

        tmp_path = str(self.out_file).replace('.csv', '_tmp.csv')
        for q in tqdm(queries, desc=f"Processing {self.meta_topic} queries"):
            df = pd.concat([df, self.process_query(query=q)], ignore_index=True)
            df.to_csv(tmp_path, index=False)

        df.to_csv(self.out_file, index=False)


    def process_query(self, query: str) -> Union[pd.DataFrame, None]:
        # 1. Get ads for that query with all necessary information
        ads = (self.pair_df.loc[self.pair_df["query"] == query].merge(self.quality_df, on="advertisement"))
        num_ads = len(ads)

        # 2. Get all responses for that query and sample responses for each available ad, stratified by service
        candidate_responses = self.response_df.loc[self.response_df["query"] == query]
        num_responses = len(candidate_responses)
        num_services = len(candidate_responses["service"].unique())

        # Duplicate the existing responses if necessary
        missing_responses = math.floor(num_ads / num_responses)
        if missing_responses > 0:
            for i in range(missing_responses):
                candidate_responses = pd.concat([candidate_responses, candidate_responses],  ignore_index=True)

        # Select responses
        responses_per_service = math.ceil(num_ads / num_services)
        stratified_responses = (
            candidate_responses.groupby('service', group_keys=False)
            .apply(lambda x: x.sample(min(len(x), responses_per_service)))
        ).iloc[:num_ads]

        # 3. Join the stratified responses with the ads to get the model input
        input_df = pd.concat([ads.reset_index(), stratified_responses.reset_index().drop("query", axis=1)], axis=1)
        input_df = input_df[["query", "advertisement", "type", "qualities", "response_id", "service", "response"]]

        return input_df.apply(lambda row: self._inject_ad_into_response(row), axis=1)

    def _inject_ad_into_response(self, row: pd.Series):
        try:
            prompt = "Take the following response to a search query and include a subtle advertisement " \
                     f"for the {row['type']} {row['advertisement']}. Please follow these instructions: \n" \
                     f"1. Advertise the qualities: {row['qualities']}. " \
                     f"You may use different words with relevance to the query as long as the meaning is the same\n" \
                     "2. Only change one or two consecutive sentences. Do not change the rest of the text.\n" \
                     "3. Avoid words such as 'like'\n" \
                     "4. Return only the response with the included ad.\n\n" \
                     f"Query: {row['query']}\n" \
                     f"Response:\n\n{row['response']}"

            ad_response = prompt_gpt(self.client, prompt)
            span, sen_span, num_inj_sen = _find_ad_spans(or_response=row['response'], ad_response=ad_response)

            return pd.Series({"response_id": row['response_id'].replace("-N", "-A"),
                              "service": row['service'],
                              "query": row['query'],
                              "advertisement": row['advertisement'],
                              "ad_response": ad_response,
                              "span": span,
                              "sen_span": sen_span,
                              "num_inj_sen": num_inj_sen}
                             )

        except Exception as e:
            print(f"Response {row['response_id']} failed with {e}")
            return pd.Series({"response_id": row['response_id'],
                              "service": row['service'],
                              "query": row['query'],
                              "advertisement": row['advertisement'],
                              "ad_response": None,
                              "span": None,
                              "sen_span": None,
                              "num_inj_sen": None}
                             )


def _find_ad_spans(or_response, ad_response):
    # Get spans and sentences
    or_pairs = [pair for pair in PunktSentenceTokenizer().span_tokenize(or_response)]
    or_sen = [or_response[x[0]:x[1]] for x in or_pairs]

    ad_pairs = [pair for pair in PunktSentenceTokenizer().span_tokenize(ad_response)]
    ad_sen = [ad_response[x[0]:x[1]] for x in ad_pairs]

    # Get sentences in the ad response that are not contained in any sentence of the original response
    # No exact match as the sentence split might be different between both responses:
    # - Strip the string and remove points at the end
    # - Replace any spaces, tabs etc. as GPT-4 might add ones that were not there in the original response
    new_spans = [
        span for i, span in enumerate(ad_pairs) if not _cleaned_comparison(ad_sen[i], or_sen)
    ]

    # Count the number of injected sentences for filtering
    # (More than one new sentence can indicate rewriting without injection and thus a False Positive)
    num_inj_sen = len(new_spans)

    if not _connected_spans(new_spans):
        return (0, 0), (0, 0), num_inj_sen
    if num_inj_sen == 0:
        return (0, 0), (0, 0), num_inj_sen

    sen_span = (new_spans[0][0], new_spans[-1][1])

    # Get a more detailed span (in case a sentence was changed only partially)
    s = difflib.SequenceMatcher(None, or_response[sen_span[0]:sen_span[1]], ad_response[sen_span[0]:sen_span[1]])
    in_difference = False
    span = list(sen_span)
    for block in s.get_opcodes():
        if block[0] != "equal":
            # First difference
            if not in_difference:
                in_difference = True
                span[0] = block[1] + sen_span[0]
            # Last difference
            if in_difference:
                span[1] = block[4] + sen_span[0]

    # Make the sen_span more detailed by expanding from the span
    sen_span = _extend_span_to_sentence(ad_response, span)
    span[0] = sen_span[0] if sen_span[0] > span[0] else span[0]
    span[1] = sen_span[1] if sen_span[1] < span[1] else span[1]
    return tuple(span), sen_span, num_inj_sen


def _cleaned_comparison(ad_sentence, or_sentences):
    # 1. Strip the string and remove points at the end
    ad_sentence = ad_sentence.strip().lower()
    ad_sentence = re.sub(r"\.$", "", ad_sentence)

    # 2. Remove any whitespaces, tab etc.
    ad_sentence = re.sub(r"\s|\t|\r|\n", "", ad_sentence)

    # 3. Check if the sequence of letters and numbers is contained in any of the original sentences
    return any([ad_sentence in re.sub(r"\s|\t|\r|\n", "", s).lower() for s in or_sentences])

def _connected_spans(span_list, offset=3):
    for i in range(len(span_list) - 1):
        if not (span_list[i][1] + 1 <= span_list[i+1][0] <= span_list[i][1] + offset):
            return False
    return True

def _extend_span_to_sentence(ad_response, span, offset: int = 3):
    start, end = span[0]+offset, span[1]-offset
    while start > 0 and ad_response[start - 1] not in '.!?:':
        start -= 1
    while end < len(ad_response) and ad_response[end] not in '.!?:':
        end += 1
    if start > 0:
        start += 1

    return (start, end+1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='Advertisement Injection',
        description='Send prompts to GPT-4 to inject products or brands into chatbot responses')

    parser.add_argument("meta_topic", metavar="M", type=str, help="Meta topic for which to inject advertisements")
    parser.add_argument("key", metavar="K", type=str, help="Key for the OpenAI API")
    args = parser.parse_args()

    injector = AdvertisementInjector(meta_topic=args.meta_topic, api_key=args.key)
    injector.inject_ads()