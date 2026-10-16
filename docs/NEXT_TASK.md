# TASK-000: восстановить воспроизводимость frozen audio baseline

## Роль

Ты — implementing Codex. Выполни только эту задачу и остановись. Следуй AGENTS.md.

## Обязательное чтение

До изменений прочитай:

1. docs/PROJECT_REQUIREMENTS.md;
2. docs/PLAN.md, только Этап 0 и общие gates;
3. docs/PROGRESS.md;
4. docs/PROJECT_INIT_STRUCTURE.md, разделы 3, 5 и 6;
5. текущий src/audio/models/audio_mamba_segment.py;
6. src/chimera_plugin.py.

## Цель

Сделать существующий audio baseline импортируемым и конфигурационно воспроизводимым из current checkout, не изменяя ни одного файла в src/audio и не начиная разработку нового fusion method.

## Разрешённые изменения

- src/fusion/__init__.py;
- src/fusion/models/__init__.py;
- src/fusion/models/av_sync_mamba_segment.py;
- src/chimera_plugin.py, только если это объективно необходимо для чистой регистрации;
- configs/wsm_mm_pd_dep_v1/audio/00_frozen_baseline.yaml;
- docs/PROGRESS.md.

Новые файлы __init__.py внутри перечисленных каталогов также разрешены.

## Источники восстановления

Исторический provider и config находятся здесь:

    logs/wsm_audio_segment_wavlm_base_l9_pool4/
      multitask_audio_mamba_2026-08-19_14-12_audio_mamba_segment_model_wsm_audio_models-e0ce-006_ea8c8d77/
        code.zip
        wsm_audio_models-e0ce-006.yaml
        summary.txt

Текущий audio module импортирует из historical provider:

- TemporalEncoder;
- _autocast_enabled;
- _head;
- _projection;
- masked_mean;
- resample_valid_sequence.

Восстанови совместимую реализацию из snapshot с минимальными изменениями. Не объявляй historical AV model новым RAMPS/fusion baseline и не добавляй его experiment config.

Canonical config должен сохранять параметры лучшего run, но:

- использовать experiment_name wsm_mm_pd_dep_v1;
- иметь понятный run_name frozen_audio_wavlm_l9_pool4;
- включать checkpoint_callback, snapshot_callback, early_stopping_callback, wsm_summary_callback и segment metrics callback;
- включать console_file_logger и mlflow_logger;
- monitor должен быть dev/mean_score, mode max;
- быть самодостаточным и не ссылаться на отсутствующий base config.

## Запрещено

- любые изменения src/audio;
- архитектурный refactor historical provider;
- перенос primitives в common в рамках этой задачи;
- запуск sweep или полного training;
- изменение data split или metric formula;
- реализация video, text, description, partial-label или нового fusion code;
- просмотр новых Test результатов;
- исправление включения Test в validation loop: это будет отдельная задача Этапа 1.

## Порядок выполнения

1. Зафиксируй git status и hash/diff состояния src/audio.
2. Извлеки только нужный historical source из code.zip и изучи saved config.
3. Добавь минимальный compatible provider и package initializers.
4. Создай canonical frozen audio config.
5. При необходимости минимально поправь plugin imports.
6. Выполни verification ниже.
7. Подтверди, что src/audio не изменён.
8. Обнови docs/PROGRESS.md, не редактируя этот файл.
9. Остановись и верни manager handoff по формату AGENTS.md.

## Acceptance criteria

- импорт src.chimera_plugin и его register завершается без warning о fusion.models;
- audio_mamba_segment_model присутствует в Chimera MODELS;
- связанные audio datamodule, loss и callbacks по-прежнему зарегистрированы;
- canonical YAML проходит доступную Chimera config validation;
- YAML содержит все обязательные callbacks/loggers и правильный monitor;
- instantiate или минимальный import smoke не требует изменения src/audio;
- git diff -- src/audio пуст;
- никакой training/Test evaluation не запускался.

## Verification

Используй фактически доступные CLI/API и запиши точные команды в PROGRESS. Как минимум проверь:

1. Python import и plugin register с warnings treated as errors для project import problems.
2. Наличие audio_mamba_segment_model и ожидаемых registry keys.
3. Chimera config validation либо, если CLI не поддерживает её для этого config, parse/resolution через официальный config API с объяснением.
4. YAML parse и assertions для experiment_name, run_name, callbacks, loggers и monitor.
5. git diff --check.
6. git diff -- src/audio должен не иметь output.

Если snapshot несовместим с установленной chimera-ml, не меняй src/audio и не маскируй проблему warning. Зафиксируй точную несовместимость как blocker.

