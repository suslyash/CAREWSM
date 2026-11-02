import os
import glob
import torch
import argparse
from tqdm import tqdm
from diarizen.pipelines.inference import DiariZenPipeline

def identify_speakers(diarization):
    """In our corpus, the speaker who speaks the most will be
    always the participant. The other, where present, the interviewer.
    """
    speaker_durations = {}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        speaker_durations[speaker] = (
            speaker_durations.get(speaker, 0.0)
            + (segment.end - segment.start)
        )

    speakers = sorted(
        speaker_durations,
        key=speaker_durations.get,
        reverse=True,
    )

    mapping = {speakers[0]: "PARTICIPANT"}
    if len(speakers) > 1:
        for speaker in speakers[1:]:
            mapping[speaker] = "INTERVIEWER"

    relabeled = diarization.rename_labels(mapping)
    return relabeled

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Diarization with DiariZen (BUTspeech)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--audio-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    wav_paths = glob.glob(os.path.join(args.audio_dir, '**', '*.wav'), recursive=True)
    diar_pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")

    for wav_path in tqdm(wav_paths):

        sample_id = os.path.basename(wav_path).split('.')[0]
        group_id, subgroup_id, profile_id, video_id = sample_id.split('__')
        output_dir = os.path.join(args.output_dir, group_id, subgroup_id, f"PROFILE_{profile_id}", f"VIDEO_{video_id}")
        os.makedirs(output_dir, exist_ok=True)
        rttm_out_path = os.path.join(output_dir, 'diarizen_diarization.rttm')

        # -- diarization
        diarization = diar_pipeline(wav_path)
        diarization = identify_speakers( diarization )
        with open(rttm_out_path, "w", encoding="utf-8") as rttm:
            diarization.write_rttm(rttm)
