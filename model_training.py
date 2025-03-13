import os
import torch
import librosa
import pandas as pd
import logging
from torch.utils.data import Dataset
from dataclasses import dataclass
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperConfig
)
from sklearn.model_selection import train_test_split
from config import CONFIG
from monitoring import RobustMonitoringCallback

logger = logging.getLogger(__name__)

class ATCDataset(Dataset):
    def __init__(self, metadata_df, processor):
        self.metadata = metadata_df
        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata.iloc[idx]
        speech, sample_rate = librosa.load(item["audio_path"], sr=CONFIG["audio_sample_rate"])
        input_features = self.processor(
            speech,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.squeeze()
        labels = self.processor.tokenizer(item["transcription"]).input_ids
        return {
            "input_features": input_features,
            "labels": labels,
            "file_path": item["audio_path"]
        }

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features):
        input_features = [feature["input_features"] for feature in features]
        labels = [feature["labels"] for feature in features]
        input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)
        labels = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding="longest",
            return_tensors="pt",
        ).input_ids
        return {"input_features": input_features, "labels": labels}

def adapt_tokenizer_for_aviation(base_tokenizer):
    from data_preparation import generate_aviation_vocabulary
    logger.info("Adapting tokenizer for aviation terminology...")
    if not os.path.exists(CONFIG["aviation_vocab_path"]):
        vocab = generate_aviation_vocabulary()
    else:
        with open(CONFIG["aviation_vocab_path"], "r") as f:
            vocab = [line.strip() for line in f.readlines()]
    new_tokens = [term for term in vocab if (" " in term or any(entity in term for entity in CONFIG["aviation_specific_entities"]))]
    new_tokens = list(set(new_tokens))
    if new_tokens:
        num_added = base_tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {num_added} aviation-specific tokens to tokenizer")
    return base_tokenizer

def train_atc_model(compute_metrics=None):
    logger.info("Starting ATC model training...")
    synthetic_metadata = pd.read_csv(os.path.join(CONFIG["atc_dataset_path"], "synthetic_metadata.csv"))
    real_metadata_path = os.path.join(CONFIG["atc_dataset_path"], "real_metadata.csv")
    if os.path.exists(real_metadata_path):
        real_metadata = pd.read_csv(real_metadata_path)
        all_metadata = pd.concat([synthetic_metadata, real_metadata], ignore_index=True)
    else:
        all_metadata = synthetic_metadata
    train_metadata, val_metadata = train_test_split(all_metadata, test_size=0.15, random_state=CONFIG["seed"])
    logger.info(f"Training on {len(train_metadata)} samples, validating on {len(val_metadata)} samples")
    config = WhisperConfig.from_pretrained(CONFIG["base_model"])
    processor = WhisperProcessor.from_pretrained(CONFIG["base_model"])
    model = WhisperForConditionalGeneration.from_pretrained(CONFIG["base_model"])
    processor.tokenizer = adapt_tokenizer_for_aviation(processor.tokenizer)
    model.resize_token_embeddings(len(processor.tokenizer))
    train_dataset = ATCDataset(train_metadata, processor)
    val_dataset = ATCDataset(val_metadata, processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    training_args = Seq2SeqTrainingArguments(
        output_dir=CONFIG["model_output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        num_train_epochs=CONFIG["num_epochs"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(CONFIG["model_output_dir"], "logs"),
        logging_steps=1,  # log more frequently (adjust as needed)
        fp16=torch.cuda.is_available(),
        seed=CONFIG["seed"],
        dataloader_num_workers=4,
        report_to="tensorboard",
        push_to_hub=False,
    )

    # Add our custom monitoring callback
    custom_callbacks = [RobustMonitoringCallback(training_args.logging_dir)]

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=custom_callbacks  # Pass the custom callback here
    )

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    model.save_pretrained(CONFIG["model_output_dir"])
    processor.save_pretrained(CONFIG["model_output_dir"])
    logger.info(f"Finished training. Model saved at {CONFIG['model_output_dir']}")
    return model, processor
