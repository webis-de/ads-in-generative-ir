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