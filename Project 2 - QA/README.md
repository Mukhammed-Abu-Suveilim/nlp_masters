# QA Project 2 — Извлечение ответов из юридических контрактов

Проект для соревнования по курсу NLP (RUDN). Задача: по тексту контракта и вопросу предсказать **точную цитату** из текста — ответ должен быть дословным фрагментом `context`.

Основной и финальный скрипт: **`qa_pipeline_v5.py`**.

---

## Задача и правила соревнования

| Параметр | Описание |
|----------|----------|
| Метрика на Kaggle | **Exact Match** — полное посимвольное совпадение предсказания и эталона |
| Данные | `train.csv` (200 строк), `test.csv` (165 строк) |
| Доп. данные | Разрешены (используем CUAD), метки теста — нет |
| Модели | Модели с курса; локальные LLM до 1B (для ранних экспериментов); API-LLM запрещены |
| Оценка | Закрытый (private) лидерборд + воспроизводимый код в одном файле |

Формат вопроса (стиль CUAD):

```
Highlight the parts (if any) of this contract related to "<тип_клаузулы>"
that should be reviewed by a lawyer. Details: <уточняющий вопрос>
```

Ответ — подстрока из `context`: от короткого имени (`Bravatek`) до длинного абзаца.

Подробнее: `Compition overview and rules.txt`, `research_brief.md`.

---

## Структура репозитория

```
QA Project 2/
├── qa_pipeline_v5.py          # Финальный пайплайн (ансамбль extractive QA)
├── pipeline_run_logs.log      # Лог полного прогона с дообучением
├── pipeline_run_v5_8_models_not_fine_tuned  # Лог прогона без дообучения
├── submission.csv             # Последний сабмит (после fine-tune)
├── submission_v5_8_models_fine_tuned.csv
├── submission_v5_8_models_not_fine_tuned.csv
├── val_metrics.json           # Метрики валидации ансамбля
├── val_predictions.csv
│
├── data/
│   ├── train.csv, test.csv, sample_submission.csv
│   └── cuad_data/             # CUAD для этапа domain adaptation
│       ├── train_separate_questions.json
│       └── test.json          # Разметка для локальной оценки теста
│
├── saved_models/              # Сохранённые веса после обучения
├── ROBERTA/                   # Ранний эксперимент с одной моделью
├── outputs/                   # Артефакты раннего LLM-пайплайна (Ollama)
├── old_code/                  # Черновики и старые версии
│
├── run_qa_pipeline.py         # Ранняя версия: Ollama qwen3.5:0.8b
├── qa_pipeline_v3.py          # Промежуточная версия
├── qa_pipeline.ipynb
├── pyproject.toml             # Окружение для LLM-экспериментов (uv)
└── research_brief.md          # Бриф для анализа стратегий улучшения
```

---

## Как работает `qa_pipeline_v5.py`

Скрипт реализует **ансамбль из 8 моделей extractive Question Answering** (BERT/RoBERTa/DeBERTa/ELECTRA/ALBERT), обученных на SQuAD/CUAD. Генеративные LLM в финальном решении не используются.

### Общая схема

```
train.csv (200) + CUAD JSON (до 2500 примеров)
        │
        ▼
┌───────────────────────────────────────┐
│  Для каждой из 8 моделей:             │
│  1. Stage 1 — domain adaptation     │  (только не-CUAD модели)
│     на выборке из CUAD                │
│  2. Stage 2 — калибровка на train     │  (160 примеров, 80/20 split)
│  3. SWA — усреднение весов            │  (последние 2 эпохи)
│  4. TTA — инференс со stride          │  (64, 128, 192)
└───────────────────────────────────────┘
        │
        ▼
 Rank-Norm Ensemble + голосование
        │
        ▼
 submission.csv (165 ответов)
```

### Модели в ансамбле

1. `akdeniz27/roberta-large-cuad`
2. `mgigena/roberta-large-cuad`
3. `deepset/deberta-v3-base-squad2`
4. `deepset/deberta-v3-large-squad2`
5. `deepset/roberta-base-squad2`
6. `deepset/electra-base-squad2`
7. `twmkn9/albert-base-v2-squad2`
8. `mrm8488/bert-small-finetuned-squadv2`

CUAD-модели уже обучены на контрактах — для них пропускается Stage 1. Остальные сначала дообучаются на ~2301 примерах из `data/cuad_data/train_separate_questions.json` (стратифицированная выборка по 41 типу клаузул).

### Ключевые техники

- **Двухэтапное обучение**: сначала CUAD (домен), затем train соревнования (калибровка).
- **Stochastic Weight Averaging (SWA)**: усреднение весов последних эпох.
- **Test-Time Augmentation (TTA)**: контекст режется с разными `stride` (64/128/192), ответ выбирается по сумме логитов start/end.
- **Rank-Norm Ensemble**: ранговая нормализация скоров моделей, взвешенное голосование; бонус если один и тот же ответ дали 2+ или 3+ модели.
- **clean_span**: постобработка границ ответа (скобки, запятые по краям).
- **Умное возобновление**: если модель уже есть в `saved_models/`, обучение пропускается.

### Режимы (`MODE` в начале файла)

| Режим | Действие |
|-------|----------|
| `"full"` | Обучение + валидация + тест + сабмит |
| `"val_only"` | Только валидация |
| `"infer_only"` | Только инференс (при `SKIP_TRAINING = True`) |

### Выходные файлы

| Файл | Содержимое |
|------|------------|
| `submission.csv` | Сабмит для Kaggle: `id`, `answers` |
| `val_predictions.csv` | Предсказания на валидации с gold/F1/EM |
| `val_metrics.json` | Сводные метрики валидации |
| `pipeline_run.log` | Подробный лог (создаётся при запуске) |
| `saved_models/<имя_модели>/` | Сохранённые веса |

### Запуск

```powershell
cd "QA Project 2"
pip install pandas torch transformers datasets scikit-learn accelerate
python qa_pipeline_v5.py
```

Нужен GPU (в логах — ~84 GB VRAM свободно на используемой карте). Полный прогон с обучением всех 8 моделей занимает несколько часов.

---

## Ранние эксперименты

### Ollama + qwen3.5:0.8b (`run_qa_pipeline.py`)

Первый подход: локальная LLM, очистка текста, keyword-retrieval (~4000 символов), промпт, выравнивание ответа через `rapidfuzz`.

**Результат**: validation EM = **2.5%** (1 из 40). Подход оказался непригоден для strict exact match на длинных контрактах.

Артефакты: `outputs/`, `research_brief.md`.

### Одна RoBERTa (`ROBERTA/`)

Отдельный прогон одной модели: validation EM = **15%** (`ROBERTA/val_metrics.json`).

---

## Результаты из логов

### Прогон с дообучением (финальный)

Источник: `pipeline_run_logs.log` (2026-06-04, 8 моделей, Stage 1 + Stage 2 + SWA + TTA + ensemble).

| Метрика | Значение | Примечание |
|---------|----------|------------|
| Validation EM | **62.5%** | 80/20 split train (40 примеров), strict strip-сравнение |
| Validation F1 | **73.1%** | Нормализованный F1 (как в SQuAD) |
| Локальный test EM | **83.03%** | По `data/cuad_data/test.json`, 165/165 совпадений |
| Локальный test F1 | **90.78%** | Нормализованный F1 |

Файл сабмита: `submission.csv` / `submission_v5_8_models_fine_tuned.csv`.

### Прогон без дообучения (только предобученные веса + ensemble)

Источник: `pipeline_run_v5_8_models_not_fine_tuned`.

| Метрика | Значение |
|---------|----------|
| Локальный test EM | **72.73%** |
| Локальный test F1 | **78.98%** |

Файл сабмита: `submission_v5_8_models_not_fine_tuned.csv`.

### Важно про метрики

- **Kaggle** считает **строгий exact match** (посимвольно, без нормализации SQuAD).
- **Локальная оценка в скрипте** для test.json использует **нормализованный EM** (нижний регистр, без артиклей и пунктуации) — он обычно **выше**, чем на лидерборде.
- Validation EM в скрипте (62.5%) ближе к формату соревнования, чем локальный test EM (83%).

---


## Зависимости

**Финальный пайплайн (`qa_pipeline_v5.py`):**

```
pandas, numpy, torch, transformers, datasets, scikit-learn, accelerate
```

**Ранний LLM-пайплайн (`run_qa_pipeline.py`):**

```
uv sync   # см. pyproject.toml: pandas, scikit-learn, rapidfuzz, ollama, jupyter
```

---

## Краткая история версий

| Версия | Подход | Validation EM |
|--------|--------|---------------|
| `run_qa_pipeline.py` | Ollama qwen3.5:0.8b + retrieval | 2.5% |
| `ROBERTA/` | Одна RoBERTa-large-CUAD | 15% |
| `qa_pipeline_v3.py` | Промежуточный ансамбль | — |
| **`qa_pipeline_v5.py`** | **8 моделей + CUAD + SWA + TTA + ensemble** | **62.5%** (val), **83%** (локальный test, норм.) |

---

## Полезные ссылки внутри проекта

- `Compition overview and rules.txt` — официальные правила
- `pipeline_run_logs.log` — полный лог финального обучения
