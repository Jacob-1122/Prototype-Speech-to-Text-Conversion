import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import logging
import torch
from sklearn.model_selection import train_test_split
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from config import CONFIG

logger = logging.getLogger(__name__)

def compute_aviation_specific_metrics(predictions, references):
    import re
    from difflib import SequenceMatcher

    def extract_callsigns(text):
        return re.findall(r'\b[A-Z]{2,4}\d{1,4}\b', text)

    atc_commands = {"cleared", "takeoff", "land", "contact", "taxi", "hold", "climb", "descend",
                    "approach", "departure", "report", "squawk", "line", "position", "vector", "turn"}

    def extract_commands(text):
        words = text.lower().split()
        return [word for word in words if word in atc_commands]

    def extract_numeric(text):
        return re.findall(r'\b\d+\b', text)

    def accuracy_metric(pred_tokens, ref_tokens):
        if not ref_tokens:
            return 1.0
        correct = sum(1 for token in ref_tokens if token in pred_tokens)
        return correct / len(ref_tokens)

    callsign_accuracies = []
    command_accuracies = []
    numeric_accuracies = []

    for pred, ref in zip(predictions, references):
        pred_callsigns = extract_callsigns(pred)
        ref_callsigns = extract_callsigns(ref)
        callsign_accuracies.append(accuracy_metric(pred_callsigns, ref_callsigns))
        pred_commands = extract_commands(pred)
        ref_commands = extract_commands(ref)
        command_accuracies.append(accuracy_metric(pred_commands, ref_commands))
        pred_numerics = extract_numeric(pred)
        ref_numerics = extract_numeric(ref)
        numeric_accuracies.append(accuracy_metric(pred_numerics, ref_numerics))

    avg_callsign = sum(callsign_accuracies) / len(callsign_accuracies) if callsign_accuracies else 0.0
    avg_command = sum(command_accuracies) / len(command_accuracies) if command_accuracies else 0.0
    avg_numeric = sum(numeric_accuracies) / len(numeric_accuracies) if numeric_accuracies else 0.0

    return {
        "callsign_accuracy": avg_callsign,
        "command_accuracy": avg_command,
        "numeric_accuracy": avg_numeric
    }

def evaluate_atc_model(model=None, processor=None, test_split=0.1):
    logger.info("Evaluating ATC model...")
    if model is None or processor is None:
        logger.info("Loading saved model...")
        model = WhisperForConditionalGeneration.from_pretrained(CONFIG["model_output_dir"])
        processor = WhisperProcessor.from_pretrained(CONFIG["model_output_dir"])
    synthetic_metadata = pd.read_csv(os.path.join(CONFIG["atc_dataset_path"], "synthetic_metadata.csv"))
    real_metadata_path = os.path.join(CONFIG["atc_dataset_path"], "real_metadata.csv")
    if os.path.exists(real_metadata_path):
        real_metadata = pd.read_csv(real_metadata_path)
        all_metadata = pd.concat([synthetic_metadata, real_metadata], ignore_index=True)
    else:
        all_metadata = synthetic_metadata
    _, test_metadata = train_test_split(all_metadata, test_size=test_split, random_state=CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    results = {"file_path": [], "reference": [], "prediction": [], "wer": [], "difficulty": []}
    for _, row in test_metadata.iterrows():
        try:
            import librosa
            speech, sample_rate = librosa.load(row["audio_path"], sr=CONFIG["audio_sample_rate"])
            input_features = processor(
                speech,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features.to(device)
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            from jiwer import wer
            error_rate = wer(row["transcription"], prediction)
            results["file_path"].append(row["audio_path"])
            results["reference"].append(row["transcription"])
            results["prediction"].append(prediction)
            results["wer"].append(error_rate)
            results["difficulty"].append(row["difficulty"])
        except Exception as e:
            logger.error(f"Error processing {row['audio_path']}: {str(e)}")
            continue
    results_df = pd.DataFrame(results)
    overall_wer = results_df["wer"].mean()
    difficulty_metrics = results_df.groupby("difficulty")["wer"].mean()
    aviation_metrics = compute_aviation_specific_metrics(
        results_df["prediction"].tolist(),
        results_df["reference"].tolist()
    )
    results_path = os.path.join(CONFIG["model_output_dir"], "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    report = {
        "overall_wer": overall_wer,
        "difficulty_wer": difficulty_metrics.to_dict(),
        "callsign_accuracy": aviation_metrics["callsign_accuracy"],
        "command_accuracy": aviation_metrics["command_accuracy"],
        "numeric_accuracy": aviation_metrics["numeric_accuracy"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_test_samples": len(results_df)
    }
    report_path = os.path.join(CONFIG["model_output_dir"], "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    plt.figure(figsize=(10, 6))
    plt.hist(results_df["wer"], bins=20, alpha=0.7)
    plt.xlabel("Word Error Rate (WER)")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Word Error Rates")
    plt.savefig(os.path.join(CONFIG["model_output_dir"], "wer_distribution.png"))
    plt.figure(figsize=(10, 6))
    difficulty_metrics.plot(kind="bar")
    plt.xlabel("Difficulty Level")
    plt.ylabel("Average Word Error Rate (WER)")
    plt.title("WER by Difficulty Level")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["model_output_dir"], "wer_by_difficulty.png"))
    logger.info(f"Evaluation completed. Results saved to {results_path}")
    logger.info(f"Overall WER: {overall_wer:.4f}")
    return report
