import os

CONFIG = {
    "base_model": "openai/whisper-tiny",  # You can change to 'openai/whisper-tiny' if needed.
    "aviation_vocab_path": os.path.join("data", "aviation_vocabulary.txt"),
    "atc_dataset_path": os.path.join("data", "atc_dataset"),
    "model_output_dir": os.path.join("models", "atc_whisper"),
    "audio_sample_rate": 16000,
    "max_audio_length": 30,
    "batch_size": 2,
    "learning_rate": 3e-5,
    "num_epochs": 1,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2,
    "seed": 42,
    "live_stream_url": "https://s1-fmt2.liveatc.net/kjfk9_s",
    "airports": ["KJFK", "KLAX", "KORD", "KATL", "KEWR", "KBOS", "KDFW"],
    "aviation_specific_entities": ["runway", "taxiway", "altimeter", "squawk", "approach", "departure"]
}
