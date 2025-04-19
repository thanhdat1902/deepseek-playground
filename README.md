# Synthetic Fire Disaster Analysis Dataset

## Overview

This project provides a high-quality synthetic dataset of wildfire disaster events generated through advanced prompt engineering. Designed for use in machine learning, natural language processing (NLP), and emergency response research, the dataset includes structured entries simulating real-world fire incidents with detailed analytical breakdowns.

The generation process leverages state-of-the-art large language models (LLMs), including OpenAI's GPT, Anthropic Claude, and DeepSeek, and focuses exclusively on fire-related disaster scenarios.

---

## Objectives

- 📌 Create a structured dataset for training, benchmarking, and evaluation of models in:

  - Fire Disaster response automation
  - Summarization and reasoning
  - Risk assessment and situational analysis

- 🧠 Design and validate an optimized prompt that produces high-fidelity synthetic fire incident analyses

- 📂 Deliver 100+ synthetic examples in standardized `.jsonl` format for compatibility with modern ML frameworks

---

## Files Included

| File                     | Description                                                              |
| ------------------------ | ------------------------------------------------------------------------ |
| `fire_llm_dataset.json`  | List of synthetic fire incidents in standard JSON format                 |
| `fire_llm_dataset.jsonl` | Line-delimited version for use with NLP pipelines and model training     |
| `prompt.txt`             | Final engineered prompt for generating structured fire incident analyses |
| `README.md`              | Project overview, usage instructions, and format documentation           |

---

## Dataset Format

Each `.jsonl` entry consists of two fields:

- `instruction`: Fire event metadata formatted as a semi-structured input
- `output`: LLM-generated analysis in a consistent template with labeled sections

### Sample Entry

```json
{
  "instruction": "event_title: Wildfire in Sierra Foothills\nevent_description: ...",
  "output": "<disaster_analysis>\n\n1. **Key Data Points Extraction:**\n   - Event Title: Wildfire in Sierra Foothills\n   - Disaster Type: Wildfire\n   - ...\n</disaster_analysis>"
}
```

---

## Prompt Usage

To generate new fire disaster analyses:

1. Open the prompt in `prompt.txt`
2. Replace the example event details with new parameters (e.g., location, coordinates, burned area, weather data)
3. Input the modified prompt into an LLM (e.g., GPT-4, Claude)
4. Save the response in the `.jsonl` format to expand your dataset

---

## Applications

This dataset and methodology can be applied to:

- Fine-tuning and evaluating disaster-aware LLMs
- Training summarization, classification, and QA models
- Testing emergency response systems under simulated conditions
- Enhancing situational awareness in digital twin environments

---

## Quality Assurance

- 🔍 Output verified for structural consistency and completeness
- ✅ Prompt tested on multiple LLM platforms
- 📊 Data conforms to analysis requirements across 13 detailed categories (e.g., weather impact, uncertainty quantification, infrastructure effects)
