import os
import time
import logging

# Optional: Force CPU-only by uncommenting the following line:
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Monkey-patch Accelerator to remove unexpected keyword (if needed)
try:
    from accelerate import Accelerator

    original_accelerator_init = Accelerator.__init__


    def new_accelerator_init(self, *args, **kwargs):
        kwargs.pop("dispatch_batches", None)
        original_accelerator_init(self, *args, **kwargs)


    Accelerator.__init__ = new_accelerator_init
except Exception as e:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("atc_model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from setup_directories import setup_directories
from performance import log_performance_usage
from data_preparation import generate_aviation_vocabulary, generate_synthetic_atc_data
from database import initialize_transcription_database  # Updated import here!
from model_training import train_atc_model
from evaluation import evaluate_atc_model


# Uncomment the following line if you want to use live transcription:
# from live_transcription import ATCLiveTranscriber

def main():
    logger.info("Starting ATC Whisper Model Training and Evaluation")

    # Ensure all directories exist.
    setup_directories()

    step_start = time.time()
    generate_aviation_vocabulary()
    log_performance_usage("Aviation Vocabulary Generation", step_start)

    step_start = time.time()
    generate_synthetic_atc_data(num_samples=10)
    log_performance_usage("Synthetic Data Generation", step_start)

    step_start = time.time()
    initialize_transcription_database()
    log_performance_usage("Database Initialization", step_start)

    step_start = time.time()
    model, processor = train_atc_model()
    log_performance_usage("Model Training", step_start)

    step_start = time.time()
    evaluation_report = evaluate_atc_model(model, processor)
    log_performance_usage("Model Evaluation", step_start)

    # Uncomment below to run a live transcription demo:
    # step_start = time.time()
    # transcriber = ATCLiveTranscriber()
    # for result in transcriber.start_live_transcription(duration=60):
    #     print(f"[{result['start']:.1f}s - {result['end']:.1f}s] {result['text']}")
    # log_performance_usage("Live Transcription Demo", step_start)

    logger.info("ATC Whisper Model Pipeline Completed Successfully")


if __name__ == "__main__":
    main()
