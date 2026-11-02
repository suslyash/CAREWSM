from textgrid import TextGrid

def extract_participant_turns(rttm_path, max_gap=3.0, min_turn_duration=0.3):
    """
        Identify participant conversation turns, provided that
    silence gaps between diarization sub-segments are smaller than max_gap and
    that the interviewers doesn't interacts.
    """

    segments = []

    # -- reading rttm diarization
    with open(rttm_path, "r") as f:

        for line in f:
            parts = line.strip().split()

            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]

            segments.append( (speaker, start, end) )

    segments.sort(key=lambda x: x[1]) # just in case

    # -- identifying participant turns
    turn_times = []

    current_start = None
    current_end = None

    for speaker, start, end in segments:

        # -- interviewer => hard boundary
        if speaker == "INTERVIEWER":

            if current_start is not None:
                turn_times.append(
                    (current_start, current_end)
                )

                current_start = None
                current_end = None

            continue

        # -- participant speech
        if current_start is None:
            current_start = start
            current_end = end
        else:
            gap = start - current_end

            if gap <= max_gap:
                current_end = end
            else:
                turn_times.append(
                    (current_start, current_end)
                )

                current_start = start
                current_end = end

    if current_start is not None:
        turn_times.append(
            (current_start, current_end)
        )

    turn_times = [
        (start, end)
        for start, end in turn_times
        if (end - start) >= min_turn_duration
    ]

    return turn_times

def transcript_substrings_from_textgrid(turn_times, textgrid_path):
    """
    Assigns WhisperX words to participant turns using the midpoint rule.

    Parameters
    ----------
    turn_times : list of (start, end)
        Output of extract_participant_turns().
    textgrid_path : str
        Path to the WhisperX TextGrid.

    Returns
    -------
    turn_transcripts : list of str
        One transcript per turn.
    """

    tg = TextGrid.fromFile(textgrid_path)

    # Assuming the first tier contains the words
    tier = tg[0]

    # Read all words
    words = []

    for interval in tier:

        word = interval.mark.strip()

        # Ignore empty intervals
        if not word:
            continue

        words.append({
            "mid": (interval.minTime + interval.maxTime) / 2,
            "word": word
        })

    # Assign words to turns
    word_idx = 0
    n_words = len(words)
    turn_transcripts = []

    for turn_id, (turn_start, turn_end) in enumerate(turn_times, start=1):

        current_words = []

        # Skip words before the current turn
        while (
            word_idx < n_words
            and words[word_idx]["mid"] < turn_start
        ):
            word_idx += 1
        start_word_idx = word_idx

        # Collect words inside the turn
        while (
            word_idx < n_words
            and words[word_idx]["mid"] <= turn_end
        ):
            current_words.append(words[word_idx]["word"])
            word_idx += 1
        end_word_idx = word_idx

        turn_transcripts.append(
            (
                turn_id,
                turn_start, turn_end,
                start_word_idx, end_word_idx,
                " ".join(current_words).strip(),
            )
        )

    return turn_transcripts
