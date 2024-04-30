# Detecting Advertisements in Generative IR
This repository contains code to 
1. [Collect](ads_in_generative_ir/collect) responses from conversational search systems for a given set of [queries](ads_in_generative_ir/resources/queries)   
2. [Generate](ads_in_generative_ir/collect) versions of the original responses with a selection of [advertisements](ads_in_generative_ir/resources/advertisements)
3. [Classify](ads_in_generative_ir/classify) responses into containing ads or not using sentence transformers

## Structure
### Collection
In the collection step, responses are obtained using the queries in the resource folder. 
Here is an example command for getting banking-related responses from Bing.
```
python ads_in_generative_ir/collect/extraction.py banking -s=bing
```
The collected responses can then be filtered for having a specific lang and belonging to the top-k queries (500 in the example below):
```
python ads_in_generative_ir/collect/filtering.py banking -l=en -q=500
```

### Generation
The generation step takes filtered responses and injects advertisements into them.
1. Suggest products/brands to advertise. You should manually filter the results before proceeding
```
python ads_in_generative_ir/generate/suggest_advertisements.py banking [OPENAI-KEY]
```
2. Select products/brands for each query. Again, adapt the results manually
```
python ads_in_generative_ir/generate/select_advertisements.py banking [OPENAI-KEY]
```
3. Inject advertisements into responses based on the query-ad-pairs created in the previous step
```
python ads_in_generative_ir/generate/inject_advertisements.py banking [OPENAI-KEY]
```

### Classification
The final step is to create a dataset and use it to train sentence transformers.
1. Create a dataset of responses, ensuring that advertisements are not leaked across splits and query overlap is minimized
```
python ads_in_generative_ir/classify/response_ds.py
```
2. Create a dataset of sentence pairs from the original response dataset
```
python ads_in_generative_ir/classify/sentence_ds.py
```
3. Train the models (An optional hold-out topic can be provided with the `-t` flag. Otherwise, the default test set is used)
```
python ads_in_generative_ir/classify/train.py all-mpnet-base-v2
```
4. Evaluate the models on the test data (Again, a hold-out topic can be defined with `-t)
```
python ads_in_generative_ir/classify/evaluate.py all-mpnet-base-v2
```

### LLM-Based Detection
In addition to fine-tuning pre-trained sentence transformers, we applied a set of instruction-tuned LLMs to the task of detecting advertisements.
The code and utilized prompts can be found in the `detect_advertisement_MODEL.py` for each model in [detect](ads_in_generative_ir/detect).

One example is 
```
python ads_in_generative_ir/detect/detect_advertisements_gpt4.py test [OPENAI-KEY]
```

## Setup
A suitable conda environment can be created from the `.lock`-file as follows:
```
conda create --name ads --file conda-linux-64.lock
conda activate ads
poetry install
```

# Paper
## Abstract

Conversational search engines such as YouChat and Microsoft Copilot use large language models (LLMs) to generate responses to queries. It is only a small step to also let the same technology insert ads within the generated responses - instead of separately placing ads next to a response. Inserted ads would be reminiscent of native advertising and product placement, both of which are very effective forms of subtle and manipulative advertising. Considering the high computational costs associated with LLMs, for which providers need to develop sustainable business models, users of conversational search engines may very well be confronted with generated native ads in the near future. In this paper, we thus take a first step to investigate whether LLMs can also be used as a countermeasure, i.e., to block generated native ads. We compile the Webis Generated Native Ads 2024 dataset of queries and generated responses with automatically inserted ads, and evaluate whether LLMs or fine-tuned sentence transformers can detect the ads. In our experiments, the investigated LLMs struggle with the task but sentence transformers achieve precision and recall values above 0.9.

## Citation
```
@InProceedings{schmidt:2024,
  author =                   {Sebastian Schmidt and Ines Zelch and Janek Bevendorff and Benno Stein and Matthias Hagen and Martin Potthast},
  booktitle =                {WWW '24: Proceedings of the ACM Web Conference 2024},
  doi =                      {10.1145/3589335.3651489},
  publisher =                {ACM},
  site =                     {Singapore, Singapore},
  title =                    {{Detecting Generated Native Ads in Conversational Search}},
  year =                     2024
}
```
