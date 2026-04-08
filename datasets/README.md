# Datasets for "Finding Variables Under Consideration"

This directory contains datasets for studying which variables LLMs consider when generating responses, measured via perplexity changes. The focus is on pragmatic implicature, scalar diversity, and datasets with responses varying along specific axes.

---

## Directory Structure

```
datasets/
├── bigbench_implicatures/          # BIG-bench implicature task
├── UCL-DARK_ludwig/                # Ruis et al. (LUDWIG) + do-pigs-fly CSVs
├── tasksource_implicatures/        # tasksource/implicatures (George & Mamidi 2020)
├── lm-pragmatics/                  # Hu et al. 2022 pragmatics benchmark
├── scalar_diversity/               # Scalar diversity items (langcog + Van Tiel)
├── scalar_adjectives/              # Garí Soler & Apidianaki 2020 (DEMELO/CROWD/WILKINSON)
├── stanford_politeness/            # Stanford Politeness Corpus
├── PUB_pragmatics/                 # PUB: Pragmatics Understanding Benchmark
└── README.md (this file)
```

---

## Dataset Details

### 1. BIG-bench Implicatures (`bigbench_implicatures/`)

**Source:** Google BIG-bench benchmark  
**Paper:** BIG-bench collaboration (2022)  
**GitHub:** https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/implicatures  
**Files:**
- `task.json` — 492 dialogue examples; task is to predict whether Speaker 2's response means "yes" or "no"
- `README.md` — Task description

**Format:** JSON with `input` (dialogue) and `target_scores` ({yes: 0/1, no: 0/1})  
**Size:** 492 examples  
**Relevance:** Direct test of conversational implicature interpretation

**Re-download:**
```bash
curl -L -o bigbench_implicatures/task.json \
  "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/implicatures/task.json"
```

---

### 2. LUDWIG / do-pigs-fly — Conversational Implicature (`UCL-DARK_ludwig/`)

**Source:** UCL-DARK lab (LauraRuis et al.)  
**Paper:** "Do Large Language Models know what humans know?" — Ruis et al. 2022  
**HuggingFace:** https://huggingface.co/datasets/UCL-DARK/ludwig  
**GitHub:** https://github.com/LauraRuis/do-pigs-fly  
**Files:**
- `test_conversational_implicatures.csv` — 629 test examples (from do-pigs-fly repo)
- `dev_conversational_implicatures.csv` — Dev set examples
- `type_labels.csv` — Linguistic type labels for each implicature
- `alignment_prompt_templates.csv` — Prompt templates
- `test_0shot.parquet` — HF parquet (0-shot test split, 600 examples)
- `validation_0shot.parquet` — HF parquet (0-shot validation, 118 examples)

**CSV Format:** `Context utterance, Response utterance, Implicature`  
**Size:** 629 test examples  
**Relevance:** Core benchmark for conversational implicature — utterance + response → implied yes/no

**Re-download CSVs:**
```bash
BASE="https://raw.githubusercontent.com/LauraRuis/do-pigs-fly/main/data"
curl -L -o UCL-DARK_ludwig/test_conversational_implicatures.csv "$BASE/test_conversational_implicatures.csv"
curl -L -o UCL-DARK_ludwig/dev_conversational_implicatures.csv "$BASE/dev_conversational_implicatures.csv"
```

**Re-download HF parquet:**
```bash
curl -L -o UCL-DARK_ludwig/test_0shot.parquet \
  "https://huggingface.co/api/datasets/UCL-DARK/ludwig/parquet/0-shot/test/0.parquet"
```

---

### 3. tasksource/implicatures (`tasksource_implicatures/`)

**Source:** tasksource project (Sileo 2023), based on George & Mamidi 2020  
**Paper:** "Conversational implicatures in English dialogue" — George & Mamidi (2020)  
**HuggingFace:** https://huggingface.co/datasets/tasksource/implicatures  
**Files:**
- `train.parquet` — 1,000 examples with correct/incorrect implicatures

**Format:** `context, response, correct_implicature, incorrect_implicature`  
**Size:** 1,000 examples  
**License:** GPL  
**Relevance:** Conversational implicature NLI — given context+response, identify correct implied meaning

**Re-download:**
```bash
curl -L -o tasksource_implicatures/train.parquet \
  "https://huggingface.co/api/datasets/tasksource/implicatures/parquet/default/train/0.parquet"
```

---

### 4. LM-Pragmatics (`lm-pragmatics/`)

**Source:** Hu et al. 2022  
**Paper:** "A fine-grained comparison of pragmatic language understanding in humans and language models" — Hu, Floyd, Jouravlev, Fedorenko, Gibson (2022)  
**ArXiv:** https://arxiv.org/abs/2212.06801  
**HuggingFace:** https://huggingface.co/datasets/clembench-playpen/lm-pragmatics  
**Files (seed_0 split for each config):**
- `coherence_seed0.parquet` — Story coherence (200 rows total)
- `maxims_seed0.parquet` — Gricean maxim violations (95 rows total) ← most relevant for implicature
- `indirect_speech_seed0.parquet` — Indirect speech acts (100 rows total)
- `irony_seed0.parquet` — Irony understanding (125 rows total)

**Note:** This is NOT the same as Hu et al. 2024 (PUB, arXiv 2401.07078). See PUB below.  
**Relevance:** Tests 12 pragmatic phenomena including maxims (Gricean implicature), irony, indirect speech

**Re-download:**
```bash
for config in coherence maxims indirect_speech irony; do
  curl -L -o "lm-pragmatics/${config}_seed0.parquet" \
    "https://huggingface.co/api/datasets/clembench-playpen/lm-pragmatics/parquet/${config}/seed_0/0.parquet"
done
```

---

### 5. Scalar Diversity (`scalar_diversity/`)

**Source:** Multiple  
**Files:**
- `van_tiel_2014_scalar_pairs.csv` — 43 scalar pairs from Van Tiel et al. 2016 with SI rates
- `langcog_e1_items.csv` — Scalar implicature sentence pairs from LangCog lab (Frank & Goodman)
- `langcog_e1_results_key.csv` — Worker ID mapping for results

#### Van Tiel et al. 2016 Scalar Diversity
**Paper:** "Scalar diversity" — Van Tiel, Van Miltenburg, Zevakhina, Geurts (2016)  
**Journal:** Journal of Semantics, DOI: 10.1093/jos/ffu017  
**Format:** `lower_term, upper_term, SI_rate, scale_type, example_sentence`  
**Size:** 43 scalar pairs with implicature endorsement rates  
**Relevance:** Key reference for scalar diversity — different scalar pairs have very different SI rates; useful for probing which scalar alternatives LLMs consider

#### LangCog Scalar Implicature Items
**Source:** Frank & Goodman lab (https://github.com/langcog/scalar_implicature)  
**Format:** `scale, domain, manip, plurality, sent1, sent2`  
**Scales:** all/some, always/sometimes, and/or, two/three, good/excellent, like/love  
**Relevance:** Naturalistic sentence pairs with scalar contrasts

**Re-download Van Tiel data:** See paper appendix; reconstructed from Table 1 of Van Tiel et al. 2016.

---

### 6. Scalar Adjectives — DEMELO / CROWD / WILKINSON (`scalar_adjectives/`)

**Source:** Garí Soler & Apidianaki (2020)  
**Paper:** "Scalar Adjective Identification and Scalar Rank Prediction" — Garí Soler & Apidianaki (EMNLP 2020, NAACL 2021)  
**GitHub:** https://github.com/ainagari/scalar_adjs  
**Files:**
- `scalar_adjectives_summary.csv` — Summary of all 187 pairs across all three datasets
- `demelo/terms/*.terms` — 87 scalar adjective scales (adjective rankings)
- `demelo/gold_rankings/*.rankings` — 87 gold scalar rankings
- `crowd/terms/*.terms` — 79 scalar adjective scales
- `crowd/gold_rankings/*.rankings` — 79 gold rankings
- `wilkinson/terms/*.terms` — 21 scalar adjective scales
- `wilkinson/gold_rankings/*.rankings` — 21 gold rankings

**Format (`.terms`):** Tab-separated `rank\tadjective` (lower number = weaker)  
**Format (`.rankings`):** Tab-separated `rank\tadjective`  
**Dataset origins:**
- **DEMELO**: De Melo & Bansal (2013) — crowdsourced scalar rankings
- **CROWD**: Rubinstein et al. (2013) — crowd-annotated adjective scales
- **WILKINSON**: Wilkinson (1996) — expert-curated adjective scales

**Size:** 187 scalar adjective pairs total (87 + 79 + 21)  
**Relevance:** Scalar adjective pairs for probing LLM representations of scalar strength

**Re-download:**
```bash
python3 -c "
import urllib.request, json, os
def dl_dir(repo, path, local):
    os.makedirs(local, exist_ok=True)
    url = f'https://api.github.com/repos/{repo}/contents/{path}'
    with urllib.request.urlopen(url) as r:
        items = json.loads(r.read())
    for item in items:
        if item['type'] == 'file':
            with urllib.request.urlopen(item['download_url']) as r:
                open(os.path.join(local, item['name']), 'wb').write(r.read())
        elif item['type'] == 'dir':
            dl_dir(repo, f\"{path}/{item['name']}\", os.path.join(local, item['name']))
for ds in ['demelo','crowd','wilkinson']:
    for sub in ['terms','gold_rankings']:
        dl_dir('ainagari/scalar_adjs', f'data/{ds}/{sub}', f'scalar_adjectives/{ds}/{sub}')
"
```

---

### 7. Stanford Politeness Corpus (`stanford_politeness/`)

**Source:** Danescu-Niculescu-Mizil et al. 2013; Cleanlab repackaging  
**Paper:** "A computational approach to politeness with application to social factors" (2013)  
**HuggingFace:** https://huggingface.co/datasets/Cleanlab/stanford-politeness  
**Files:**
- `train.csv` — Training set (~2K examples)
- `test.csv` — Test set (~500 examples)

**Format:** `prompt, completion` where completion ∈ {polite, neutral, impolite}  
**Relevance:** Responses varying along politeness axis — useful for testing whether LLMs consider politeness as a variable

**Re-download:**
```bash
BASE="https://huggingface.co/datasets/Cleanlab/stanford-politeness/resolve/main/fine-tuning"
curl -L -o stanford_politeness/train.csv "$BASE/train.csv"
curl -L -o stanford_politeness/test.csv "$BASE/test.csv"
```

---

### 8. PUB — Pragmatics Understanding Benchmark (`PUB_pragmatics/`)

**Source:** Doshi et al. 2024 (IIT Bombay / cfilt)  
**Paper:** "PUB: A Pragmatics Understanding Benchmark for Assessing LLMs' Pragmatics Capabilities" — Hu et al. 2024  
**ArXiv:** https://arxiv.org/abs/2401.07078  
**HuggingFace:** https://huggingface.co/datasets/cfilt/PUB  
**GitHub:** https://github.com/meetdoshi90/PUB  
**Files:**
- `task_1_sample.jsonl` — 100 examples from Task 1 (implicature: direct vs. indirect answers; 2500 total)
- `task_14_deixis.jsonl` — All 458 examples from Task 14 (deixis/reference)
- `task_1.zip` — Full Task 1 zip (2500 examples, 681KB uncompressed)
- `task_14.zip` — Full Task 14 zip (458 examples)

**Format:** `{"pretext": "...", "options": [...], "correct answer": "...", "id": "..."}`  
**Total dataset:** 28,000 examples across 14 tasks covering implicature, presupposition, reference, deixis  
**Relevance:** 14 pragmatic reasoning tasks; Task 1 specifically tests direct vs. indirect speech acts

**Re-download:**
```bash
for task in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  curl -L -o "PUB_pragmatics/task_${task}.zip" \
    "https://huggingface.co/datasets/cfilt/PUB/resolve/main/data/task_${task}.zip"
done
```

---

## Quick Summary Table

| Dataset | Size | Task Type | Key Variable | Source |
|---------|------|-----------|-------------|--------|
| BIG-bench Implicatures | 492 | Yes/No | Conversational implicature | BIG-bench 2022 |
| LUDWIG / do-pigs-fly | 629 test | Yes/No | Conversational implicature | Ruis et al. 2022 |
| tasksource/implicatures | 1,000 | NLI | Implicature identification | George & Mamidi 2020 |
| LM-Pragmatics | 1,365 | MC | 12 pragmatic phenomena | Hu et al. 2022 |
| Van Tiel Scalar Diversity | 43 pairs | Annotation | Scalar implicature rates | Van Tiel et al. 2016 |
| LangCog Scalar Items | 108 pairs | MC | Scalar quantifiers/adjectives | Frank & Goodman |
| Scalar Adjectives (DEMELO/CROWD/WILKINSON) | 187 pairs | Ranking | Adjective scalar strength | Garí Soler & Apidianaki 2020 |
| Stanford Politeness | ~2,500 | Classification | Politeness level | Danescu-Niculescu-Mizil 2013 |
| PUB Pragmatics | 28,000 | MC | 14 pragmatic phenomena | Doshi et al. 2024 |

---

## Notes on Missing Datasets

- **Van Tiel et al. 2014 raw experimental data**: Not publicly available on GitHub/OSF. The `van_tiel_2014_scalar_pairs.csv` in `scalar_diversity/` reconstructs the key Table 1 data (43 pairs with SI rates) from the published paper. For the full experimental data, contact the authors or check the Journal of Semantics supplementary materials.

- **GYAFC (Grammarly's Yahoo Answers Formality Corpus)**: Restricted access; requires agreement with Yahoo Terms of Service. Available at: https://github.com/raosudha89/GYAFC-corpus

- **PUB full dataset**: Only Tasks 1 and 14 are extracted to JSONL. To extract other tasks, unzip `task_N.zip` and read the `.jsonl` file inside.
