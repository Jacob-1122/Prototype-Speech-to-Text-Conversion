import os
import random
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import logging
from config import CONFIG

logger = logging.getLogger(__name__)

def generate_aviation_vocabulary():
    logger.info("Generating aviation vocabulary...")
    atc_phrases = [
        "cleared for takeoff", "cleared to land", "cleared for the approach", "cleared as filed",
        "cleared direct to", "cleared to cross", "hold short", "line up and wait", "position and hold",
        "taxi to", "contact", "monitor", "squawk", "ident", "report", "maintain", "climb and maintain",
        "descend and maintain", "expedite climb", "expedite descent", "turn left heading",
        "turn right heading", "fly heading", "intercept", "join", "proceed", "hold at",
        "wilco", "unable", "say again", "standby", "roger", "affirm", "negative",
        "ceiling", "visibility", "RVR", "wind", "altimeter", "QNH", "temperature", "dew point",
        "ILS", "visual approach", "RNAV approach", "GPS approach", "missed approach", "go around",
        "departure", "SID", "STAR", "transition", "vectors", "final approach course",
        "runway", "taxiway", "ramp", "gate", "apron", "terminal", "hold short line",
        "DME", "VOR", "NDB", "ATIS", "AWOS", "ASOS", "TRACON", "ARTCC", "TMA", "CTA",
        "mayday", "pan pan", "emergency", "declaring", "priority", "minimum fuel", "fuel emergency"
    ]
    airlines = ["AAL", "UAL", "DAL", "SWA", "JBU", "ASA", "FFT", "SKW", "HAL", "FDX", "UPS"]
    callsigns = []
    for airline in airlines:
        for i in range(10):
            flight_num = random.randint(1, 9999)
            callsigns.append(f"{airline}{flight_num}")
    for i in range(20):
        prefix = random.choice(["N", "C-", "G-"])
        if prefix == "N":
            suffix = f"{random.randint(1, 999)}{random.choice(['A','B','C','D','E','F','G'])}{random.choice(['X','Y','Z'])}"
        else:
            suffix = f"{random.randint(1, 9999)}{random.choice(['A','B','C','D','E','F','G'])}"
        callsigns.append(f"{prefix}{suffix}")
    runways = []
    for num in range(1, 37):
        rwy_num = f"{num:02d}"
        runways.extend([f"runway {rwy_num}", f"runway {rwy_num} left", f"runway {rwy_num} right", f"runway {rwy_num} center"])
    taxiways = [f"taxiway {letter}" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    navaids = []
    consonants = "BCDFGHJKLMNPQRSTVWXYZ"
    vowels = "AEIOU"
    for _ in range(50):
        if random.random() < 0.5:
            waypoint = "".join(random.choice(consonants) + random.choice(vowels) for _ in range(2))[:-1]
        else:
            waypoint = "".join(random.choice(consonants + vowels) for _ in range(3))
        navaids.append(waypoint)
    altitudes = []
    for alt in range(1, 18):
        altitudes.append(f"{alt} thousand")
        if alt <= 4:
            for hundred in range(1, 10):
                altitudes.append(f"{alt} thousand {hundred} hundred")
    flight_levels = [f"flight level {fl}" for fl in range(180, 451, 10)]
    headings = [f"heading {hdg:03d}" for hdg in range(0, 360, 10)]
    speeds = [f"speed {spd}" for spd in range(160, 351, 10)]
    frequencies = []
    for whole in range(118, 137):
        for decimal in range(0, 10, 5):
            frequencies.append(f"{whole} point {decimal}")
            if decimal == 0:
                frequencies.append(f"{whole} decimal {decimal}")
    transponders = [f"squawk {code:04d}" for code in random.sample(range(1000, 7778), 50)]
    all_vocab = set(atc_phrases + callsigns + runways + taxiways + navaids +
                    altitudes + flight_levels + headings + speeds + frequencies + transponders)
    with open(CONFIG["aviation_vocab_path"], "w") as f:
        for item in sorted(all_vocab):
            f.write(f"{item}\n")
    logger.info(f"Generated {len(all_vocab)} aviation vocabulary terms")
    return list(all_vocab)

def add_radio_static(audio, static_level=0.05):
    static = np.random.normal(0, static_level, len(audio))
    return audio + static

def simulate_interruption(audio, max_gaps=2):
    audio_out = audio.copy()
    num_gaps = random.randint(1, max_gaps)
    for _ in range(num_gaps):
        gap_start = random.randint(0, len(audio) - 1)
        gap_duration = random.randint(int(0.05 * len(audio)), int(0.2 * len(audio)))
        gap_end = min(gap_start + gap_duration, len(audio))
        gap_type = random.choice(["silence", "static", "distortion"])
        if gap_type == "silence":
            audio_out[gap_start:gap_end] = 0
        elif gap_type == "static":
            audio_out[gap_start:gap_end] = np.random.normal(0, 0.2, gap_end - gap_start)
        else:
            audio_out[gap_start:gap_end] = np.clip(audio[gap_start:gap_end] * 5, -1, 1)
    return audio_out

def apply_frequency_filter(audio, sample_rate, low_cutoff=300, high_cutoff=3000):
    from scipy import signal
    nyquist = sample_rate / 2
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(5, [low, high], btype='band')
    return signal.filtfilt(b, a, audio)

def change_speed(audio, speed_factor):
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def augment_audio_file(audio_path, output_path, transcription, difficulty_level=0):
    audio, sample_rate = librosa.load(audio_path, sr=CONFIG["audio_sample_rate"])
    audio = apply_frequency_filter(audio, sample_rate)
    if difficulty_level >= 1:
        audio = add_radio_static(audio, static_level=0.02 * difficulty_level)
    if difficulty_level >= 2:
        speed_factor = random.uniform(0.9, 1.1)
        audio = change_speed(audio, speed_factor)
    if difficulty_level >= 3:
        audio = simulate_interruption(audio, max_gaps=difficulty_level - 2)
    sf.write(output_path, audio, sample_rate)
    if difficulty_level >= 4 and len(transcription) > 20:
        words = transcription.split()
        num_words_to_remove = int(0.1 * len(words) * (difficulty_level - 3))
        for _ in range(num_words_to_remove):
            if words:
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)
        transcription = " ".join(words)
    return transcription

def generate_synthetic_atc_data(num_samples=10):
    logger.info(f"Generating {num_samples} synthetic ATC samples...")
    if not os.path.exists(CONFIG["aviation_vocab_path"]):
        vocab = generate_aviation_vocabulary()
    else:
        with open(CONFIG["aviation_vocab_path"], "r") as f:
            vocab = [line.strip() for line in f.readlines()]
    atc_templates = [
        "{callsign} cleared to land runway {runway}",
        "{callsign} climb and maintain {altitude}",
        "{callsign} descend and maintain {altitude}",
        "{callsign} turn left heading {heading}",
        "{callsign} turn right heading {heading}",
        "{callsign} contact {facility} on {frequency}",
        "{callsign} squawk {transponder}",
        "{callsign} taxi to runway {runway} via {taxiway}",
        "{callsign} hold short of runway {runway}",
        "{callsign} cleared for takeoff runway {runway}, fly heading {heading}",
        "{callsign} traffic alert, {traffic_position}",
        "{callsign} winds {wind_direction} at {wind_speed}, altimeter {altimeter}",
        "{callsign} expect vectors for the {approach_type} approach runway {runway}",
        "{facility} radio check",
        "{callsign} report {reporting_point}",
        "{facility} visibility {visibility} ceiling {ceiling}",
        "{callsign} maintain visual separation from traffic",
        "{callsign} expedite climb through {altitude} for traffic",
        "{callsign} go around, traffic on the runway",
        "{callsign} caution wake turbulence, preceding traffic {preceding_traffic}"
    ]
    callsigns = [v for v in vocab if any(airline in v for airline in CONFIG["airports"] + ["AAL", "UAL", "DAL", "N"])]
    runways = [v for v in vocab if v.startswith("runway")]
    taxiways = [v for v in vocab if v.startswith("taxiway")]
    altitudes = [v for v in vocab if "thousand" in v or "flight level" in v]
    headings = [v for v in vocab if v.startswith("heading")]
    frequencies = [v for v in vocab if "point" in v or "decimal" in v]
    transponders = [v for v in vocab if v.startswith("squawk")]
    facilities = ["tower", "ground", "approach", "departure", "center"]
    traffic_positions = ["12 o'clock", "2 o'clock", "10 o'clock", "3 miles", "5 miles", "on final"]
    wind_directions = [f"{dir:03d}" for dir in range(0, 360, 10)]
    wind_speeds = [f"{speed} knots" for speed in range(5, 31, 5)]
    altimeter_settings = [f"{alt:04d}" for alt in range(2900, 3100)]
    approach_types = ["ILS", "RNAV", "visual", "GPS", "VOR"]
    reporting_points = ["established", "final", "outer marker", "ATIS", "field in sight", "on the numbers"]
    visibility_values = [f"{vis} miles" for vis in [1, 2, 3, 5, 7, 10]] + ["unlimited"]
    ceiling_values = [f"{ceil} feet" for ceil in [500, 1000, 1500, 2000, 3000, 5000]] + ["clear"]
    preceding_traffic = ["heavy Boeing", "Airbus", "regional jet", "light aircraft"]
    synthetic_audio_dir = os.path.join(CONFIG["atc_dataset_path"], "synthetic")
    os.makedirs(synthetic_audio_dir, exist_ok=True)
    metadata = []
    for i in range(num_samples):
        template = random.choice(atc_templates)
        filled_template = template.format(
            callsign=random.choice(callsigns) if "{callsign}" in template else "",
            runway=random.choice(runways).replace("runway ", "") if "{runway}" in template else "",
            altitude=random.choice(altitudes) if "{altitude}" in template else "",
            heading=random.choice(headings).replace("heading ", "") if "{heading}" in template else "",
            facility=random.choice(facilities) if "{facility}" in template else "",
            frequency=random.choice(frequencies) if "{frequency}" in template else "",
            transponder=random.choice(transponders).replace("squawk ", "") if "{transponder}" in template else "",
            taxiway=random.choice(taxiways).replace("taxiway ", "") if "{taxiway}" in template else "",
            traffic_position=random.choice(traffic_positions) if "{traffic_position}" in template else "",
            wind_direction=random.choice(wind_directions) if "{wind_direction}" in template else "",
            wind_speed=random.choice(wind_speeds) if "{wind_speed}" in template else "",
            altimeter=random.choice(altimeter_settings) if "{altimeter}" in template else "",
            approach_type=random.choice(approach_types) if "{approach_type}" in template else "",
            reporting_point=random.choice(reporting_points) if "{reporting_point}" in template else "",
            visibility=random.choice(visibility_values) if "{visibility}" in template else "",
            ceiling=random.choice(ceiling_values) if "{ceiling}" in template else "",
            preceding_traffic=random.choice(preceding_traffic) if "{preceding_traffic}" in template else ""
        )
        audio_path = os.path.join(synthetic_audio_dir, f"synthetic_{i:04d}.wav")
        sample_rate = CONFIG["audio_sample_rate"]
        duration = random.uniform(1.5, 5.0)
        samples = int(duration * sample_rate)
        t = np.linspace(0, duration, samples, False)
        fundamental = 120
        audio = np.zeros_like(t)
        for harmonic in range(1, 6):
            audio += (1.0 / harmonic) * np.sin(2 * np.pi * fundamental * harmonic * t)
        envelope = np.ones_like(t)
        attack = int(0.02 * samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        release = int(0.1 * samples)
        envelope[-release:] = np.linspace(1, 0, release)
        audio = audio * envelope
        audio = audio / np.max(np.abs(audio))
        sf.write(audio_path, audio, sample_rate)
        difficulty = random.randint(0, 4)
        augmented_path = os.path.join(synthetic_audio_dir, f"synthetic_{i:04d}_aug.wav")
        augmented_transcription = augment_audio_file(audio_path, augmented_path, filled_template, difficulty)
        metadata.append({
            "audio_path": augmented_path,
            "transcription": augmented_transcription,
            "original_transcription": filled_template,
            "difficulty": difficulty,
            "duration": duration,
            "type": "synthetic"
        })
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(CONFIG["atc_dataset_path"], "synthetic_metadata.csv"), index=False)
    logger.info(f"Generated {len(metadata)} synthetic samples")
    return metadata_df
