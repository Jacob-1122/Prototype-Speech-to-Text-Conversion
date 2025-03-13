import os
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
import logging
from faster_whisper import WhisperModel
from config import CONFIG

logger = logging.getLogger(__name__)

class ATCLiveTranscriber:
    """Real-time ATC transcription system."""
    def __init__(self, model_path=None):
        self.model_path = model_path or CONFIG["model_output_dir"]
        self.running = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.buffer = []
        self.sample_rate = CONFIG["audio_sample_rate"]
        logger.info(f"Loading model from {self.model_path}...")
        # Use GPU if available and not forced off via CUDA_VISIBLE_DEVICES
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            if torch.cuda.is_available():
                self.model = WhisperModel("medium", device="cuda", compute_type="float16")
            else:
                self.model = WhisperModel("medium", device="cpu", compute_type="int8")
        else:
            self.model = WhisperModel("medium", device="cpu", compute_type="int8")
        with open(CONFIG["aviation_vocab_path"], "r") as f:
            self.vocab = [line.strip() for line in f.readlines()]

    def start_live_transcription(self, input_device=None, duration=300):
        self.running = True
        audio_thread = threading.Thread(target=self._record_audio, args=(input_device, duration))
        audio_thread.daemon = True
        audio_thread.start()
        transcription_thread = threading.Thread(target=self._transcribe_audio)
        transcription_thread.daemon = True
        transcription_thread.start()
        try:
            while self.running:
                if not self.result_queue.empty():
                    result = self.result_queue.get()
                    yield result
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False
            logger.info("Stopping live transcription...")
        audio_thread.join(timeout=1)
        transcription_thread.join(timeout=1)

    def _record_audio(self, input_device, duration):
        chunk_duration = 5
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            device=input_device,
            callback=self._audio_callback
        )
        logger.info(f"Recording from device {input_device} for {duration} seconds...")
        with stream:
            end_time = time.time() + duration
            while self.running and time.time() < end_time:
                time.sleep(0.1)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio stream status: {status}")
        self.buffer.append(indata.copy())
        if len(self.buffer) * indata.shape[0] >= self.sample_rate * 2:
            audio_data = np.concatenate(self.buffer)
            self.audio_queue.put(audio_data)
            self.buffer = []

    def _transcribe_audio(self):
        while self.running:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                try:
                    segments, info = self.model.transcribe(
                        audio_data,
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    for segment in segments:
                        transcription = self._post_process_transcription(segment.text)
                        self.result_queue.put({
                            "text": transcription,
                            "start": segment.start,
                            "end": segment.end,
                            "confidence": segment.avg_logprob
                        })
                except Exception as e:
                    logger.error(f"Error transcribing audio: {str(e)}")
            else:
                time.sleep(0.1)

    def _post_process_transcription(self, text):
        from difflib import SequenceMatcher
        for term in self.vocab:
            if " " not in term and len(term) < 5:
                continue
            words = text.split()
            for i in range(len(words)):
                if i < len(words):
                    word = words[i]
                    ratio = SequenceMatcher(None, word.lower(), term.lower()).ratio()
                    if ratio > 0.8:
                        words[i] = term
                if i < len(words) - 1:
                    phrase = words[i] + " " + words[i + 1]
                    ratio = SequenceMatcher(None, phrase.lower(), term.lower()).ratio()
                    if ratio > 0.8:
                        words[i] = term
                        words[i + 1] = ""
            text = " ".join(word for word in words if word)
        return text

    def stop(self):
        self.running = False
