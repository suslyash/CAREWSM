from pathlib import Path

root = Path("../logs/wsm_audio_segment_wavlm_base_l9_pool4")
output = Path("all_summary_2nd_stage.txt")

with output.open("w", encoding="utf-8") as out:
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue

        if exp_dir.name == "_sweeps":
            continue

        summary = exp_dir / "summary.txt"
        if not summary.is_file():
            continue

        # Ищем yaml-конфиг с любым именем
        yaml_files = sorted([
            *exp_dir.glob("*.yaml"),
            *exp_dir.glob("*.yml"),
        ])

        out.write(f"EXPERIMENT: {exp_dir.name}\n")

        # Конфиг сразу после названия эксперимента
        if yaml_files:
            for yaml_file in yaml_files:
                out.write(f"CONFIG: {yaml_file.name}\n")
                out.write(yaml_file.read_text(encoding="utf-8").rstrip())
                out.write("\n")
        else:
            out.write("CONFIG: NOT FOUND\n")

        out.write("RESULTS:\n")
        content = summary.read_text(encoding="utf-8")
        out.write(content.rstrip())
        out.write("\n\n")

print(f"Saved to {output.resolve()}")