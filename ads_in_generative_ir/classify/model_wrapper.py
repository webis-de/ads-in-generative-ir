from abc import ABC, abstractmethod
from datasets import concatenate_datasets, DatasetDict
import os
import torch as T
from torchmetrics import Accuracy, AUROC, F1Score
from transformers import AutoTokenizer

from ads_in_generative_ir import PROJECT_PATH

# ================================================================
# Preparation
# ================================================================
DATA_PATH = PROJECT_PATH / 'data'
MODEL_OUT_PATH = PROJECT_PATH / 'models'

class ModelWrapper(ABC):
    def __init__(self, sbert_model: str, disjoint_col: str = "advertisement", holdout_topic: str = None,
                 model_suffix: str = None,
                 dataset_prefix: str = "sentence_pairs"):
        """
        Abstract class. Used to coordinate models from ads_in_generative_ir.models
        :param sbert_model:         Name to identify a sentence transformer model on Huggingface
        :param disjoint_col:        Column that is disjoint across train, test, and validation split.
                                    Determines which dataset will be used for training
        :param holdout_topic:       Optional: Name of a meta topic to remove from train and validation set and use as
                                    the test set. Data from the test set for non-holdout topics will be moved to the
                                    train set.
        :param model_suffix:        Optional: Suffix for the model_name. The base name is constructed from other input
        :param dataset_prefix:      "sentence_pairs" or "responses". The full name is constructed from the other input
        """
        self.disjoint_col = disjoint_col
        self.holdout_topic = holdout_topic

        # Construct the dataset_name and model_name, model_name
        self.dataset_name = f"{dataset_prefix}_{disjoint_col}_disjoint.hf"
        model_name = f"{disjoint_col}_{sbert_model}"
        if holdout_topic:
            model_name += f"_{holdout_topic}"
        if model_suffix:
            model_name += f"_{model_suffix}"

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.loss = T.nn.BCEWithLogitsLoss()

        print("Preparing the tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{sbert_model}")

        # Set the metrics:
        self.metrics = {"accuracy": Accuracy(task="binary").to(self.device),
                        "auroc": AUROC(task="binary").to(self.device),
                        "f1": F1Score(task="binary").to(self.device)}

        # Defining the path
        self.weight_path = MODEL_OUT_PATH / f"weights/{model_name}.pt"
        self.log_path = MODEL_OUT_PATH / f"tensorboard_logs/{model_name}/"
        self.csv_path = MODEL_OUT_PATH / f"csv_logs/{model_name}.csv"

        print("Initial preparation completed.")
        print(f"- Path to model weights: {self.weight_path}")
        print(f"- Tensorboard logs: {self.log_path}")
        print(f"- CSV logs: {self.csv_path}")

        # Ensure that the weights and logs folder exist
        self._create_model_folders()

    @staticmethod
    def _create_model_folders():
        if not os.path.exists(MODEL_OUT_PATH):
            os.mkdir(MODEL_OUT_PATH)

        subfolder_list = ['weights', 'tensorboard_logs', 'csv_logs']
        for folder in subfolder_list:
            path = MODEL_OUT_PATH / folder

            if not os.path.exists(path):
                os.mkdir(path)

    # ----------------------------------------------------------------
    # Dataset preparation
    # ----------------------------------------------------------------
    @abstractmethod
    def _load_dataset(self):
        """
        Function to load a local HuggingFace dataset. Sets self.dataset attribute
        """

    def _distribute_ds_by_holdout(self):
        if not hasattr(self, 'dataset'):
            self._load_dataset()

        # Determine values for the disjoint column to ensure no leakage when redistributing data
        holdout_train_x = set(
            [v for v in
             self.dataset["train"].filter(lambda x: self._is_holdout(element=x))[self.disjoint_col]]
        )
        train_x = set(
            [v for v in
             self.dataset["train"].filter(lambda x: self._is_not_holdout(element=x))[self.disjoint_col]]
        )
        holdout_val_x = set(
            [v for v in
             self.dataset["validation"].filter(lambda x: self._is_holdout(element=x))[self.disjoint_col]]
        )
        val_x = set(
            [v for v in
             self.dataset["validation"].filter(lambda x: self._is_not_holdout(element=x))[self.disjoint_col]]
        )
        holdout_x = [x for x in [*holdout_train_x, *holdout_val_x] if x not in [*train_x, *val_x]]

        # Redistribute the data (the test set is disjoint from train and val and thus does not need to be tested)
        # The only source of leakage is from one split (train/val) having shared values on the disjoint col for
        # different meta topics
        test_ds = concatenate_datasets([
            self.dataset["train"].filter(lambda x: self._move_to_test(element=x, holdout_x=holdout_x)),
            self.dataset["validation"].filter(lambda x: self._move_to_test(element=x, holdout_x=holdout_x)),
            self.dataset["test"].filter(lambda x: x["meta_topic"] == self.holdout_topic)
        ])
        train_ds = concatenate_datasets([
            self.dataset["train"].filter(lambda x: self._keep_in_split(element=x, holdout_x=holdout_x)),
            self.dataset["test"].filter(lambda x: x["meta_topic"] != self.holdout_topic)
        ])
        val_ds = self.dataset["validation"].filter(lambda x: self._keep_in_split(element=x, holdout_x=holdout_x))

        self.dataset = DatasetDict({
            'train': train_ds,
            'test': test_ds,
            'validation': val_ds
        })

    # Functions for redistributing by a holdout column
    def _is_holdout(self, element):
        return (element["meta_topic"] == self.holdout_topic) & (element[self.disjoint_col] is not None)

    def _is_not_holdout(self, element):
        return (element["meta_topic"] != self.holdout_topic) & (element[self.disjoint_col] is not None)

    def _move_to_test(self, element, holdout_x):
        return ((element["meta_topic"] == self.holdout_topic)
                & (element[self.disjoint_col] in holdout_x or element[self.disjoint_col] is None))

    def _keep_in_split(self, element, holdout_x):
        return ((element["meta_topic"] != self.holdout_topic)
                & (element[self.disjoint_col] not in holdout_x))

    @abstractmethod
    def tokenize_data(self):
        """
        Function to tokenize the dataset. Should set self.train_ds, self.val_ds, and self.test_ds based on use case
        """

    @abstractmethod
    def _tokenize_function(self, example):
        """
        Function to tokenize an individual example. Returns results of the tokenizer.
        """

    # ----------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------
    def loss_epoch(self, model, dataloader, optimizer=None):
        """
        Function to calculate the loss for epoch
        :param model:           The model used in the epoch
        :param dataloader:      The dataloader to obtain batched data
        :param optimizer:       Optional: The optimizer to update the weights when in training
        :return:                The loss value for the entire epoch (normalized by the number of data points)
        """

        # Reset the loss at the beginning of each epoch
        ep_loss = 0.0
        ds_len = len(dataloader.dataset)

        # Initialize empty tensors to store labels and logits for metric calculation
        epoch_labels = T.empty(size=(0, 1), device=self.device, dtype=T.int32)
        epoch_logits = T.empty(size=(0, 1), device=self.device, dtype=T.float32)

        # Loop over all batches in the data
        for batch in dataloader:
            # Get the labels
            labels = batch['labels']

            # Get the logits from the batch
            logits = model(batch=batch)

            # Update epoch tensors
            epoch_labels = T.cat((epoch_labels, labels), 0)
            epoch_logits = T.cat((epoch_logits, logits), 0)

            # Compute the loss value based on the labels and logits; Optimizer is passed in case of usage with train
            loss_val = self.loss_batch(logits=logits, labels=labels, optimizer=optimizer)

            # Update the running loss
            ep_loss += loss_val

        # Get the epoch values for all the metrics
        epoch_metrics = self.metrics_epoch(logits=epoch_logits, labels=epoch_labels)

        # Return the normalized loss and the metrics
        return (ep_loss / ds_len), epoch_metrics


    def loss_batch(self, logits, labels, optimizer=None):
        """
        Function to calculate the loss on one batch
        :param logits:          The logits of the current batch
        :param labels:          The labels for each of the sentence pairs
        :param optimizer:       Optional: The optimizer to update the weights when in training
        :return:                The loss value for the batch
        """

        labels = labels.float()
        loss = self.loss(logits, labels)

        if optimizer is not None:
            # Reset the gradients, compute new ones and perform a weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()


    def metrics_epoch(self, logits, labels):
        """
        Function to calculate the metrics for the current epoch
        :param logits:      The logits of the current epoch
        :param labels:      The labels for each of the sentence pairs
        :return:            The metric values in a dictionary
        """

        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(logits, labels)

        return results