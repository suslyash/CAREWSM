import sys
import warnings
if not hasattr(sys, "warn"):
    sys.warn = warnings.warn
import os
import sys
import glob
import librosa
import tempfile
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
from distutils.util import strtobool
from audio_utils import get_turn_times, sample_output_dir

import warnings
warnings.filterwarnings("ignore")

PATH_TO_DISVOICE_REPO="./Disoice/" # change accordingly to your setup!
sys.path.insert(1, PATH_TO_DISVOICE_REPO)
from disvoice.prosody import Prosody
from disvoice.phonation import Phonation
from disvoice.articulation import Articulation

SAMPLING_RATE = 16000
# -- after checking official DisVoice documentation
HOP_SEC = {"prosody": 0.01, "phonation": 0.02, "articulation": 0.04}
FRAME_OFFSET = {"prosody": 0, "phonation": 11, "articulation": 2}

def extract_disvoice(wav_path, is_static=True):
    feature_id = "static" if is_static else "dynamic"
    output_dir = sample_output_dir(wav_path, args.output_dir, args.metadata_csv)
    expected_outputs = [os.path.join(output_dir, f"disvoice_{name}_{feature_id}.csv") for name in ("prosody", "phonation", "articulation")]
    if all(os.path.exists(path) for path in expected_outputs):
        return

    # -- additional metadata setup
    meta_cols = ["turn_id", "turn_start_sec", "turn_end_sec"]
    if feature_id == "dynamic":
        meta_cols += ["frame_start_sec", "frame_end_sec"]

    waveform, _ = librosa.load(wav_path, sr=SAMPLING_RATE, mono=True)

    # -- path setup
    sample_id = os.path.basename(wav_path).split('.')[0]
    output_dir = sample_output_dir(wav_path, args.output_dir, args.metadata_csv)

    # -- assuming speaker diarization is already processed and following same output dir structure

    rows_prosody = []
    rows_phonation = []
    rows_articulation = []
    turn_times = get_turn_times(wav_path, output_dir, len(waveform) / SAMPLING_RATE, use_diarization=args.use_diarization)

    turn_waveforms = []
    for start_sec, end_sec in turn_times:

        start_sample = int(start_sec * SAMPLING_RATE)
        end_sample = int(end_sec * SAMPLING_RATE)

        turn_waveforms.append(
            waveform[start_sample:end_sample]
        )


    for turn_id, (chunk, (start_sec, end_sec)) in enumerate( zip(turn_waveforms, turn_times), start=1):
        signal = chunk.astype(np.float32)
        duration_sec = len(signal) / SAMPLING_RATE

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, signal, SAMPLING_RATE)

            try:
                prosody_feats = prosody.extract_features_file(tmp.name, static=is_static, fmt='csv')
            except IndexError:
                prosody_feats = pd.DataFrame(np.full((1, 103), np.nan)) if is_static else pd.DataFrame(np.full((1, 13), np.nan))
                print(f"Warning: no voiced frames in {wav_path}")

            try:
                phonation_feats = phonation.extract_features_file(tmp.name, static=is_static, fmt="csv")
            except Exception as exc:
                print(f"Warning: phonation failed for {wav_path}: {exc}")
                phonation_feats = pd.DataFrame(np.full((1, 28 if is_static else 7), np.nan))

            try:
                articulation_feats = articulation.extract_features_file(tmp.name, static=is_static, fmt="csv")
            except Exception as exc:
                print(f"Warning: articulation failed for {wav_path}: {exc}")
                articulation_feats = pd.DataFrame(np.full((1, 488 if is_static else 58), np.nan))

        def enrich_feats(features, subset_id):
            features["turn_id"] = turn_id
            features["turn_start_sec"] = start_sec
            features["turn_end_sec"] = end_sec

            if feature_id == 'dynamic':
                frame_idx = np.arange(len(features)) + FRAME_OFFSET[subset_id]
                features["frame_start_sec"] = start_sec + frame_idx * HOP_SEC[subset_id]
                features["frame_end_sec"] = features["frame_start_sec"] + HOP_SEC[subset_id]

            return features[meta_cols + [c for c in features.columns if c not in meta_cols]]

        prosody_feats = enrich_feats(prosody_feats, subset_id="prosody")
        phonation_feats = enrich_feats(phonation_feats, subset_id="phonation")
        articulation_feats = enrich_feats(articulation_feats, subset_id="articulation")

        rows_prosody.append(prosody_feats)
        rows_phonation.append(phonation_feats)
        rows_articulation.append(articulation_feats)

    df = pd.concat(rows_prosody, ignore_index=True)
    output_path = os.path.join(output_dir, f"disvoice_prosody_{feature_id}.csv")
    df.to_csv(output_path, index=False)

    df = pd.concat(rows_phonation, ignore_index=True)
    output_path = os.path.join(output_dir, f"disvoice_phonation_{feature_id}.csv")
    df.to_csv(output_path, index=False)

    df = pd.concat(rows_articulation, ignore_index=True)
    output_path = os.path.join(output_dir, f"disvoice_articulation_{feature_id}.csv")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='DisVoice-based Feature Extraction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--audio-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--metadata-csv', type=str, default=None)
    parser.add_argument('--use-diarization', action='store_true')
    parser.add_argument('--is-static', type=lambda x: bool(strtobool(x)), default=True)
    args = parser.parse_args()

    # -- feature extractor build-up
    prosody = Prosody()
    phonation = Phonation()
    articulation = Articulation()

    # -- speech feature extraction
    wav_paths = glob.glob(os.path.join(args.audio_dir, '**/segments', '*.wav'), recursive=True)
    for wav_path in tqdm(wav_paths):
        try:
            extract_disvoice(wav_path, is_static=args.is_static)
            extract_disvoice(wav_path, is_static=not args.is_static)
        except Exception as exc:
            print(f"[ERROR] {wav_path}: {type(exc).__name__}: {exc}", flush=True)

