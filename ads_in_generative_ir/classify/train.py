import argparse
from datasets import load_from_disk
import os
import pandas as pd
import torch as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ads_in_generative_ir import PROJECT_PATH
import ads_in_generative_ir.classify.models as m
from ads_in_generative_ir.classify.model_wrapper import ModelWrapper

# ================================================================
# Preparation
# ================================================================
DATA_PATH = PROJECT_PATH / 'data'

# ================================================================
# Main class
# ================================================================
class TrainingWrapper(ModelWrapper):
    def __init__(self,  sbert_model: str, disjoint_col: str = "advertisement", holdout_topic: str = None,
                 model_suffix: str = None):
        """
        Wrapper to manage model training
        :param sbert_model:         Name to identify a sentence transformer model on Huggingface
        :param disjoint_col:        Column that is disjoint across train, test, and validation split.
                                    Determines which dataset will be used for training
        :param holdout_topic:       Optional: Name of a meta topic to remove from train and validation set and use as
                                    the test set. Data from the test set for non-holdout topics will be moved to the
                                    train set.
        :param model_suffix:        Optional: Suffix for the model_name. The base name is constructed from other input
        """
        super().__init__(sbert_model=sbert_model, disjoint_col=disjoint_col, holdout_topic=holdout_topic,
                         model_suffix=model_suffix, dataset_prefix="sentence_pairs")

        # Load the model as baseline from huggingface or weights from a contrastive pre-trained model.
        self.model = m.SupervisedModel(sbert_model=sbert_model)
        print("- Model prepared\n\n")


    def _load_dataset(self):
        """
        Function to load a local HuggingFace dataset. If a holdout_topic was specified, the dataset will be
        redistributed to keep all datapoints for the holdout_topic in the test split.
        Format should be:
        "id", "service", "meta_topic", "query", "advertisement", "sentence1", "sentence2", "label"
        """
        self.dataset = load_from_disk(DATA_PATH / self.dataset_name)
        if self.holdout_topic:
            self._distribute_ds_by_holdout()

    # ----------------------------------------------------------------
    # Dataset preparation
    # ----------------------------------------------------------------
    # Source: https://huggingface.co/docs/transformers/training
    def tokenize_data(self):
        """
        Function to tokenize the dataset and split it into train_ds and val_ds
        """
        if not hasattr(self, "dataset"):
            self._load_dataset()

        # Set removal columns and apply tokenization
        remove_columns = ["id", "service", "meta_topic", "query", "advertisement", "sentence1", "sentence2"]
        tokenized_ds = self.dataset.map(lambda example: self._tokenize_function(example),
                                        remove_columns=remove_columns)
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds = tokenized_ds.with_format("torch")

        # Unsqueeze the label column
        tokenized_ds = tokenized_ds.map(lambda example: {"labels": T.unsqueeze(example["labels"], dim=0)},
                                        remove_columns=["labels"])

        # Assign the splits
        self.train_ds = tokenized_ds["train"]
        self.val_ds = tokenized_ds["validation"]


    def _tokenize_function(self, example):
        # Return the tokenized strings (Note: They are in one array)
        # Pad to the maximum length of the model
        return self.tokenizer(example["sentence1"], example["sentence2"], padding="max_length", truncation=True)

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    def conduct_training(self, epochs=30, batch_size=48, optimizer_name='adam', lr=0.00005, momentum=0.7,
                         weight_decay=0, eps=1e-08, stopping_patience=5):
        """
        Function that performs training on the train_ds and validates on the val_ds.
        Checkpointing is performed based on validation loss.

        :param epochs:              How many epochs to train
        :param batch_size:          Batch size used in training

        Optimizer parameters
        ----------------------------------
        :param optimizer_name:      Used to identify the optimizer to be used
        :param lr:                  Learning rate
        :param momentum:            Momentum factor for SGD
        :param weight_decay:        Weight Decay for SGD or Adam
        :param eps:                 Epsilon for Adam

        Others
        -------------------------------------
        :param stopping_patience:  Number of epochs that val_loss is allowed to not improve before stopping
        """

        if not hasattr(self, "train_ds"):
            self.tokenize_data()

        # Prepare model and optimizer
        model = self.check_for_existing_weights(epochs=epochs)
        optimizer = m.get_optimizer(params=model.parameters(), optimizer_name=optimizer_name, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay, eps=eps)

        # Prepare early stopping
        self.stopping_patience = stopping_patience
        self.stagnant_epochs = 0

        # Create summary writer and a csv-file to write the loss values
        writer = SummaryWriter(log_dir=self.log_path)

        if not self.resume_from_existing_model:
            with open(self.csv_path, 'w') as file:
                # Fill a list with strings for the header
                out_line = ["epoch", "train_loss"]
                for name in self.metrics.keys():
                    out_line.append("train_" + name)
                out_line.append("val_loss")
                for name in self.metrics.keys():
                    out_line.append("val_" + name)

                file.write(",".join(out_line) + "\n")

        # Prepare the two dataloaders (the data is formatted for usage with torch and sent to the device)
        train_dl = DataLoader(self.train_ds.with_format("torch", device=self.device), batch_size=batch_size)
        val_dl = DataLoader(self.val_ds.with_format("torch", device=self.device), batch_size=batch_size)

        print("\nPerforming training based on the following parameters:")
        print(f"- Epochs:           {epochs}")
        print(f"- Batch size:       {batch_size}")
        print(f"- Optimizer:        {optimizer}")
        print(f"- Loss:             {self.loss}")
        print(f"- Patience:         {stopping_patience}\n\n")

        for epoch in range(self.start_epoch, epochs):
            print("\n" + "-" * 100)
            print(f"Epoch {epoch+1}/{epochs}")

            # Set the model into train mode
            model.train()
            with T.enable_grad():
                train_loss, train_metrics = self.loss_epoch(model=model, dataloader=train_dl, optimizer=optimizer)

            # Perform evaluation
            model.eval()
            with T.no_grad():
                val_loss, val_metrics = self.loss_epoch(model=model, dataloader=val_dl)

            # Change the keys in the metric-dicts to reflect whether they are from the train or val set
            for key in self.metrics.keys():
                train_metrics["train_" + key] = train_metrics.pop(key)
                val_metrics["val_" + key] = val_metrics.pop(key)

            # Print an update
            self.print_update(train_loss, val_loss, train_metrics, val_metrics)

            # Logging
            # Log the results for tensorboard and as csv
            self.logging(writer, train_loss, val_loss, train_metrics, val_metrics, epoch)

            # Perform checkpointing and check for early stopping
            if not self.continue_training_and_checkpoint(val_loss, model):
                print(f"No improvement on val_loss detected for {self.stopping_patience} epochs.")
                print("Stopping training...")
                break

        # Close the writer
        writer.flush()
        writer.close()


    def check_for_existing_weights(self, epochs):
        """
        Function to check if a model with the same weights was already trained.
        If so, the user is asked whether the training should be resumed and the necessary values are loaded.

        :param epochs:      How many epochs should be trained in total
        """
        model = self.model.to(self.device)
        self.previous_loss = float('inf')
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.resume_from_existing_model = False

        if os.path.isfile(self.csv_path):
            df = pd.read_csv(self.csv_path)
            min_row = df[df["val_loss"] == df["val_loss"].min()]

            # Determine start_epoch and ask user for confirmation
            self.start_epoch = min_row["epoch"].values[0] + 1
            previous_epochs = df.shape[0]
            if previous_epochs >= epochs:
                print(f"A model with the same name was already trained for {previous_epochs} epochs. "
                      f"Please choose a different model_name or delete the corresponding files in ./models/csv_logs, "
                      f"tensorboard_logs and weights.")
                exit(1)

            print(f"Existing logs were found under {self.csv_path}. "
                  f"Training would be resumed from epoch {self.start_epoch + 1}/{epochs}")
            resume_training = input("Would you like to continue training? (y/n): ").lower()
            if resume_training != "y":
                print("Training process aborted by user command.")
                exit(1)

            # Set loss values
            self.resume_from_existing_model = True
            self.previous_loss = self.best_val_loss = min_row["val_loss"].values[0]

            # Load the weights
            encoder_weights = T.load(self.weight_path)
            model.load_state_dict(encoder_weights)

            print(f"Existing weights loaded. Resuming training from epoch {self.start_epoch + 1}.")

        return model


    def continue_training_and_checkpoint(self, val_loss, model):
        # Initialize the return value
        continue_training = True

        # Check if an improvement to the last epoch took place; If yes, reset stagnant epochs
        if val_loss < self.previous_loss:
            self.stagnant_epochs = 0

            # Check for new optimum; If yes, update the best_val_loss and checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss


                # Checkpoint the model
                T.save(model.state_dict(), self.weight_path)
                print(f"New checkpoint for validation loss. Model weights saved to {self.weight_path}\n")

        # Otherwise increase stagnant epochs and check patience
        else:
            self.stagnant_epochs += 1

            # If no improvement took place for the specified number of epochs, stop training
            if self.stagnant_epochs > self.stopping_patience:
                continue_training = False

        # Update the previous loss
        self.previous_loss = val_loss

        return continue_training

    # ----------------------------------------------------------------
    # Logging and checkpointing
    # ----------------------------------------------------------------
    def print_update(self, train_loss, val_loss, train_metrics, val_metrics):
        """
        Function to print an update based on training
        :param train_loss:          Loss on the training data
        :param val_loss:            Loss on the validation data
        :param train_metrics:       Dictionary of metrics achieved on the training data
        :param val_metrics:         Dictionary of metrics achieved on the validation data
        """

        # Get the metrics into a string
        train_str = [str("train_loss = %10.8f | " % train_loss)]
        for name, metric in train_metrics.items():
            train_str.append(name + " = ")
            train_str.append("%10.8f | " % metric)

        val_str = [str("val_loss   = %10.8f | " % val_loss)]
        for name, metric in val_metrics.items():
            val_str.append(name + "   = ")
            val_str.append("%10.8f | " % metric)

        print("".join(train_str))
        print("".join(val_str))


    def logging(self, writer, train_loss, val_loss, train_metrics, val_metrics, epoch):
        """
        Function to perform logging for Tensorboard and into a CSV-File
        :param writer:          Instance of torch.utils.tensorboard.SummaryWriter
        :param train_loss:      Loss on the training data
        :param val_loss:        Loss on the validation data
        :param train_metrics:   Dictionary of metrics achieved on the training data
        :param val_metrics:     Dictionary of metrics achieved on the validation data
        :param epoch:           Current epoch
        :return:
        """

        # Create the outline for the CSV-File
        out_line = [str(epoch), str(train_loss)]

        # Write the losses for tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Loop over the metrics, write them for Tensorboard and append them to the out_line
        for name, metric in train_metrics.items():
            metric_item = metric.item()
            writer.add_scalar(str(name.split("_")[1] + "/train"), metric_item, epoch)
            out_line.append(str(metric_item))

        out_line.append(str(val_loss))
        for name, metric in val_metrics.items():
            metric_item = metric.item()
            writer.add_scalar(str(name.split("_")[1] + "/val"), metric_item, epoch)
            out_line.append(str(metric_item))


        with open(self.csv_path, 'a') as file:
            file.write(",".join(out_line) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Model training',
        description='Train a sentence transformer on a corpus of sentence pairs')

    parser.add_argument('sbert_model', type=str,
                        help='Name of the pretrained sentence transformer to be used as basis. E.g. all-mpnet-base-v2')
    parser.add_argument('-d', '--disjoint_col', type=str, default='advertisement',
                        help='Column for which the values are in different splits. Default value is advertisement.')
    parser.add_argument('-t', '--holdout_topic', type=str, default=None,
                        help='Name of a meta topic to remove from train and validation set and use as the test set. '
                             'Default is None.')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='Name of the optimizer used in training (adam, sgd). Default adam.')
    args = parser.parse_args()

    batch_size = 16 if args.sbert_model == 'all-mpnet-base-v2' else 48
    learning_rate = 5e-6 if args.sbert_model == 'all-mpnet-base-v2' else 1e-5

    wrapper = TrainingWrapper(sbert_model=args.sbert_model,
                              disjoint_col=args.disjoint_col,
                              holdout_topic=args.holdout_topic,
                              model_suffix=args.optimizer)
    wrapper.conduct_training(batch_size=batch_size, optimizer_name=args.optimizer, lr=learning_rate)