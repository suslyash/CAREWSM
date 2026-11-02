import os
import glob
import librosa
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
from audio_utils import get_turn_times, sample_output_dir

def process_audio(wav_path):
    # -- output path setup
    sample_id = os.path.basename(wav_path).split('.')[0]
    output_dir = sample_output_dir(wav_path, args.output_dir, args.metadata_csv)
    output_path = os.path.join(output_dir, "trillsson4_1.npz")

    if os.path.exists(output_path):
        print(f"[SKIP] Audio sample already processed: {output_path}")
        return

    # -- assuming speaker diarization is already processed and following same output dir structure

    # -- audio processing
    sr = 16000
    waveform, _ = librosa.load(wav_path, sr=sr)
    turn_times = get_turn_times(wav_path, output_dir, len(waveform) / sr, use_diarization=args.use_diarization)
    turn_waveforms = []
    for start_sec, end_sec in turn_times:

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        turn_waveforms.append(
            waveform[start_sample:end_sample]
        )


    means = []
    stds = []
    turn_ids = []
    for turn_id, (turn, (start_sec, end_sec)) in enumerate( zip(turn_waveforms, turn_times), start=1):

        try:
            input = tf.expand_dims(turn, 0)
            feats = model(input)["embedding"] # <-- (T, 1024)

            # -- pooling over time (1024,)
            mean = tf.reduce_mean(feats, axis=0)
            std = tf.math.reduce_std(feats, axis=0)

            means.append(mean.numpy())
            stds.append(std.numpy())
            turn_ids.append(turn_id)

        except RuntimeError as e:
            print(
                f"[WARNING] Failed chunk "
                f"{start_sec:.2f}-{end_sec:.2f}s: {e}"
            )
            continue

    # -- saving data
    np.savez(
        output_path,
        mean=np.stack(means),
        std=np.stack(stds),
        turn_ids=np.array(turn_ids),
    )


if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='TRILLsson4.1 Feature Embedding Extraction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--audio-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--metadata-csv', type=str, default=None)
    parser.add_argument('--use-diarization', action='store_true')
    args = parser.parse_args()

    # -- building wav2vec model
    model = hub.KerasLayer('https://tfhub.dev/google/trillsson4/1')

    wav_paths = glob.glob(os.path.join(os.path.abspath(args.audio_dir), '**/segments', '*.wav'), recursive=True)
    for wav_path in tqdm(wav_paths):
        process_audio(wav_path)
