PROJECT README & CHANGELOG

Project: Prototype Speech-to-Text Conversion
Date: [Insert Date]

OVERVIEW:
-----------
This project implements an ATC (Air Traffic Control) Whisper Model training pipeline that has been 
restructured and enhanced for stability, resource efficiency, and detailed monitoring. The pipeline 
includes data preparation, database initialization, model training, evaluation, and an optional 
live transcription module.

CHANGES MADE:
--------------
1. Model & Training Configuration:
   - Switched from "openai/whisper-medium" to "openai/whisper-tiny" to reduce memory and computation needs.
   - Reduced synthetic sample count from 500 to 10.
   - Reduced number of training epochs to 1.
   - Lowered the batch size from 8 to 2 to ease resource usage.
   
2. Directory & File Management:
   - Added a "setup_directories.py" module that automatically creates required folders:
       - data/
         - atc_dataset/ (contains synthetic and real audio files, metadata CSV, and database)
         - noise_samples/ (optional, for background noise)
         - augmented/ (optional, for augmented audio files)
       - models/
         - atc_whisper/ (where the fine-tuned model and related files are saved)
   - Updated file paths in "config.py" to ensure consistency across modules.

3. Code Modularization:
   - Split code into separate modules:
       - config.py: Global configuration parameters.
       - setup_directories.py: Directory creation and management.
       - data_preparation.py: Functions for generating aviation vocabulary and synthetic data.
       - database.py: Function for initializing the SQLite database.
       - model_training.py: Custom dataset, collator, and training function.
       - evaluation.py: Functions for evaluating the model and computing metrics.
       - live_transcription.py: Real-time transcription system.
       - monitoring.py: Custom callback (RobustMonitoringCallback) for additional TensorBoard logging.
       - performance.py: Performance logging for tracking elapsed time, CPU, and memory usage.
       - Version 3.py: Main orchestration file that calls all modules to run the complete pipeline.

4. Enhanced Monitoring & Profiling:
   - Integrated TensorBoard logging via training arguments (logging_dir set to "models/atc_whisper/logs").
   - Added a custom callback (in monitoring.py) that logs additional training metrics (e.g., gradient norms, fixed sample predictions) 
     to TensorBoard and the console.
   - Discussed use of Python's built-in cProfile module to identify performance bottlenecks.

NEXT ACTIONABLE STEPS:
------------------------
1. Run the Full Pipeline:
   - Execute "Version 3.py" to run the entire pipeline (directory setup, data preparation, training, and evaluation).
   - Verify that training completes successfully and check for any resource-related issues.

2. Monitor Training Progress:
   - Launch TensorBoard by running:
         tensorboard --logdir=models/atc_whisper/logs
     Open the provided URL (usually http://localhost:6006) in your browser.
   - In TensorBoard, check the "Scalars" tab for training loss, evaluation loss, gradient norms, and other logged metrics.
   - In the "Text" tab, review any fixed sample predictions logged at the end of each epoch.

3. Profile the Code (Optional):
   - Use cProfile to measure which functions take the most time. For example:
         import cProfile, pstats
         cProfile.run("train_atc_model()", "profile.out")
         pstats.Stats("profile.out").sort_stats('cumtime').print_stats(20)
   - Analyze the output to identify and optimize any performance bottlenecks.

4. Adjust Resource Usage if Needed:
   - If your computer crashes due to resource exhaustion:
       - Consider switching to CPU-only training by uncommenting:
             os.environ["CUDA_VISIBLE_DEVICES"] = ""
         in Version 3.py.
       - Alternatively, further reduce the batch size, number of samples, or epochs.
       - Monitor system resources (RAM, VRAM, CPU usage) using OS tools (Task Manager, NVIDIA-SMI).

5. (Optional) Integrate External Tracking Tools:
   - For even more robust experiment tracking, consider integrating Weights & Biases (wandb) to track hyperparameters, 
     metrics, and system performance over time.

6. Review Logs & Evaluation Reports:
   - Check "atc_model_training.log" for console and performance logs.
   - Open evaluation files (evaluation_report.json and evaluation_results.csv in models/atc_whisper/) to analyze model performance.
   
CONCLUSION:
--------------
This updated, modular pipeline now includes detailed logging and real-time monitoring, allowing you to see exactly what the 
model is learning and processing. Follow the actionable steps above to run the pipeline, monitor its progress, and adjust as needed.

For any further modifications or questions, refer to the individual module files.

--- End of README ---
