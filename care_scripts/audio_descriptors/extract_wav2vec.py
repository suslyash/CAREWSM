import os
import glob
import torch
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
from audio_utils import get_turn_times, sample_output_dir

def process_audio(wav_path):
    # -- output path setup
    sample_id = os.path.basename(wav_path).split('.')[0]
    output_dir = sample_output_dir(wav_path, args.output_dir, args.metadata_csv)
    output_path = os.path.join(output_dir, "wav2vec_xlsr300m.npz")

    if not os.path.exists(output_path):
        # -- assuming speaker diarization is already processed and following same output dir structure

        # -- audio processing
        waveform, sr = torchaudio.load(wav_path)
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
        waveform = waveform.to(device)
        sr = bundle.sample_rate

        turn_times = get_turn_times(wav_path, output_dir, waveform.shape[-1] / sr, use_diarization=args.use_diarization)

        means = []
        stds = []
        turn_ids = []
        for turn_id, (start_sec, end_sec) in enumerate(turn_times, start=1):
            duration = end_sec - start_sec
            start_sample = int(round(start_sec * sr))
            end_sample = int(round(end_sec * sr))
            turn = waveform[:, start_sample:end_sample]

            try:
                with torch.no_grad():
                    feats = model.extract_features(turn)[0][-1] # <-- (T, 1024)

                    # -- pooling over time (1024,)
                    mean = feats.mean(dim=1).squeeze(0)
                    std = feats.std(dim=1, unbiased=False).squeeze(0)

                    means.append(mean.cpu().numpy())
                    stds.append(std.cpu().numpy())
                    turn_ids.append(turn_id)

            except RuntimeError as e:
                if "Kernel size can't be greater than actual input size" in str(e):
                    print(
                        f"[WARNING] Skipping chunk from {wav_path}"
                        f"{start_sec:.2f}-{end_sec:.2f}s "
                        f"({turn.shape[-1]} samples)"
                    )
                    continue

                raise

        # -- saving data
        np.savez(
            output_path,
            mean=np.stack(means),
            std=np.stack(stds),
            turn_ids=np.array(turn_ids),
        )


if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='WAV2VEC-based Feature Embedding Extraction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--audio-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--metadata-csv', type=str, default=None)
    parser.add_argument('--use-diarization', action='store_true')
    args = parser.parse_args()

    # -- building wav2vec model
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_300M
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] XLS-R device: {device}', flush=True)
    model = bundle.get_model().to(device).eval()

    wav_paths = glob.glob(os.path.join(os.path.abspath(args.audio_dir), '**/segments', '*.wav'), recursive=True)
    for wav_path in tqdm(wav_paths):
        process_audio(wav_path)
