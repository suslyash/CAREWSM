# Прогресс проекта WSM

## 1. Текущее состояние

Дата инициализации плана: 2026-09-23.

Текущий этап: **документирование завершено; Этап 0 ожидает выполнения**.

Финальный Test разрешён: **нет**.

## 2. Статус этапов

| Этап | Статус | Gate | Evidence |
|---|---|---|---|
| Анализ статей и проекта | complete | Созданы baseline, structure, requirements и plan | docs/BASELINES.md, docs/PROJECT_INIT_STRUCTURE.md, docs/PROJECT_REQUIREMENTS.md, docs/PLAN.md |
| 0. Воспроизводимая основа | not started | Audio registry/config исправны, src/audio неизменён | Отсутствует |
| 1. Manifest и partial-label contract | not started | Audit и masks подтверждены | Отсутствует |
| 2. Video | not started | Не более двух models, выбран DEV winner | Отсутствует |
| 3. Text/description | not started | Не более двух models, prompt audited | Отсутствует |
| 4. Fusion baselines | not started | F0/F1/F2 сопоставимы | Отсутствует |
| 5. RAMPS | not started | Direct reliable missing-head gradient подтверждён | Отсутствует |
| 6. Ablations | not started | Claims связаны с multi-seed evidence | Отсутствует |
| 7. Final evaluation | locked | Config freeze и разрешение manager | Отсутствует |

## 3. Зафиксированный исходный baseline

Лучший найденный сохранённый audio run:

- run: wsm_audio_models-e0ce-006;
- checkpoint selection epoch: 4;
- DEV/Mean_Score: 0.787827;
- DEV depression Score: 0.747918;
- DEV Parkinson Score: 0.827735;
- TEST_NONE Mean_Score: 0.809486;
- TEST_SOFT Mean_Score: 0.815156;
- TEST_HARD Mean_Score: 0.828135;
- encoder features: WavLM-base-plus, layer 9, pool 4;
- temporal encoder: Transformer, hidden 192, 3 layers, 4 heads, 128 steps.

Эти Test values являются историческими результатами, найденными в существующем summary, а не результатом нового выбора модели. Они не разрешают использовать Test в дальнейшей разработке.

## 4. Известные блокеры

1. src/audio/models/audio_mamba_segment.py импортирует primitives из отсутствующего fusion.models.av_sync_mamba_segment.
2. Из-за этого audio_mamba_segment_model не регистрируется чисто.
3. Canonical base audio config отсутствует в configs; он существует только в logs/snapshot.
4. Текущий datamodule добавляет Test loaders в validation loop каждой эпохи.
5. Единый multilabel manifest с observed-task masks ещё не реализован.

## 5. Журнал исполнения

### DOC-001 — Анализ и исследовательский план

Статус: complete.

Изменения:

- исправлено правило root .gitignore, из-за которого docs/configs/scripts/pyproject/AGENTS не могли быть нормально версионированы;
- подготовлен анализ четырёх статей;
- описана исходная структура и найден критический broken import;
- требования переведены в проверяемые MUST/MUST NOT rules;
- предложен RAMPS и поэтапная experiment programme;
- добавлен manager/Codex orchestration contract.

Проверки:

- содержимое и headings документов просмотрены;
- исходные Python-файлы в рамках DOC-001 не редактировались;
- training и новый Test evaluation не запускались.

Следующий атомарный шаг: выполнить TASK-000 из docs/NEXT_TASK.md.

