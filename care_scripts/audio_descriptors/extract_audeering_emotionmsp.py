import os
import glob
import torch
import argparse
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

from transformers.models.wav2vec2 import Wav2Vec2Processor
from huggingface_hub import hf_hub_download
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from audio_utils import get_turn_times, sample_output_dir

SAMPLE_RATE=16000

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

def process_audio(wav_path):
    # -- output path setup
    sample_id = os.path.basename(wav_path).split('.')[0]
    output_dir = sample_output_dir(wav_path, args.output_dir, args.metadata_csv)
    embeds_output_path = os.path.join(output_dir, "emotionmsp_embeds.npz")
    trajec_output_path = os.path.join(output_dir, "emotionmsp_trajectories.csv")

    if os.path.exists(embeds_output_path) and os.path.exists(trajec_output_path):
        return

    # -- assuming speaker diarization is already processed and following same output dir structure

    # -- audio processing
    waveform, sr = torchaudio.load(wav_path)
    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    sr = SAMPLE_RATE

    turn_times = get_turn_times(wav_path, output_dir, waveform.shape[-1] / sr, use_diarization=args.use_diarization)

    embeds = []
    df_rows = []
    turn_ids = []
    for turn_id, (start_sec, end_sec) in enumerate(turn_times, start=1):
        row = {}
        duration = end_sec - start_sec

        start_sample = int(round(start_sec * sr))
        end_sample = int(round(end_sec * sr))

        turn = waveform[:, start_sample:end_sample]

        # -- normalizing input signal
        signal = processor(turn, sampling_rate=SAMPLE_RATE)
        signal = signal['input_values'][0]
        signal = signal.reshape(1, -1)
        signal = torch.from_numpy(signal).to(device)

        # run through model
        try:
            with torch.no_grad():
                embed, trajectories = model(signal)

            turn_ids.append(turn_id)
            embeds.append(embed.squeeze(0).detach().cpu().numpy())

            row["turn_id"] = turn_id
            row["turn_start_sec"] = start_sec
            row["turn_end_sec"] = end_sec
            traj = trajectories.squeeze(0)
            for idx, name in trajectory_map.items():
                row[f"{name}"] = traj[idx].detach().cpu().item()

            df_rows.append( row )

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
        embeds_output_path,
        embed=np.stack(embeds),
        turn_ids=np.array(turn_ids),
    )

    df_cols = ["turn_id", "turn_start_sec", "turn_end_sec"] + [f"{name}" for name in trajectory_map.values()]
    df = pd.DataFrame(df_rows); df = df[df_cols]
    df.to_csv(trajec_output_path, index=False)

if __name__ == "__main__":

    # -- command line arguments
    parser = argparse.ArgumentParser(description='WAV2VEC-based Feature Embedding Extraction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--audio-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--metadata-csv', type=str, default=None)
    parser.add_argument('--use-diarization', action='store_true')
    args = parser.parse_args()

    # -- building model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    checkpoint_path = hf_hub_download(model_name, "pytorch_model.bin")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = {
        key.replace(".pos_conv_embed.conv.weight_g", ".pos_conv_embed.conv.parametrizations.weight.original0")
            .replace(".pos_conv_embed.conv.weight_v", ".pos_conv_embed.conv.parametrizations.weight.original1"): value
        for key, value in checkpoint.items()
    }
    model = EmotionModel.from_pretrained(model_name, state_dict=checkpoint).to(device)

    trajectory_map = {0: "arousal", 1: "dominance", 2: "valence"}

    wav_paths = glob.glob(os.path.join(os.path.abspath(args.audio_dir), '**/segments', '*.wav'), recursive=True)
    for wav_path in tqdm(wav_paths):
        process_audio(wav_path)
