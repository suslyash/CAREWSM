import os
import glob
import librosa
import argparse
import opensmile
import numpy as np
import pandas as pd
from tqdm import tqdm
from distutils.util import strtobool

from audio_utils import get_turn_times, sample_output_dir

SAMPLING_RATE = 16000

def extract_egemaps(wav_path, is_functional=True):
    feature_id = "func" if is_functional else "lld"
    feature_level = opensmile.FeatureLevel.Functionals if is_functional else opensmile.FeatureLevel.LowLevelDescriptors

    meta_cols = ["turn_id", "turn_start_sec", "turn_end_sec"]
    if feature_id == "lld":
        meta_cols += ["frame_start_sec", "frame_end_sec"]

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=feature_level,
    )

    waveform, _ = librosa.load(wav_path, sr=SAMPLING_RATE, mono=True)

    # -- path setup
    sample_id = os.path.basename(wav_path).split('.')[0]
    output_dir = sample_output_dir(wav_path, args.output_dir, args.metadata_csv)
    output_path = os.path.join(output_dir, f"egemaps_{feature_id}.csv")

    # -- assuming speaker diarization is already processed and following same output dir structure
    turn_times = get_turn_times(wav_path, output_dir, len(waveform) / SAMPLING_RATE, use_diarization=args.use_diarization)
    turn_waveforms = []
    for start_sec, end_sec in turn_times:

        start_sample = int(start_sec * SAMPLING_RATE)
        end_sample = int(end_sec * SAMPLING_RATE)

        turn_waveforms.append(
            waveform[start_sample:end_sample]
        )

    rows = []
    for turn_id, (chunk, (start_sec, end_sec)) in enumerate( zip(turn_waveforms, turn_times), start=1):
        # -- feature extraction
        signal = chunk.astype(np.float32)
        duration_sec = len(signal) / SAMPLING_RATE
        features = smile.process_signal(signal, SAMPLING_RATE)

        features["turn_id"] = turn_id
        features["turn_start_sec"] = start_sec
        features["turn_end_sec"] = end_sec

        if feature_id == "lld":
            features["frame_start_sec"] = (
                features.index.get_level_values("start").total_seconds() + start_sec
            )

            features["frame_end_sec"] = (
                features.index.get_level_values("end").total_seconds() + start_sec
            )

        features = features[meta_cols + [c for c in features.columns if c not in meta_cols]]
        rows.append(features)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='OpenSmile-based eGeMAPS Feature Extraction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--audio-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--metadata-csv', type=str, default=None)
    parser.add_argument('--use-diarization', action='store_true')
    parser.add_argument('--is-functional', type=lambda x: bool(strtobool(x)), default=True)
    args = parser.parse_args()

    # -- speech feature extraction
    wav_paths = glob.glob(os.path.join(args.audio_dir, '**/segments', '*.wav'), recursive=True)
    for wav_path in tqdm(wav_paths):
        extract_egemaps(wav_path, is_functional=args.is_functional)
        extract_egemaps(wav_path, is_functional=not args.is_functional)
