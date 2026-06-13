# Datasets Used for HateGuard Multimodal

To achieve high accuracy across multiple languages and modalities (text and speech), HateGuard utilizes models pre-trained on a comprehensive collection of global datasets.

## 1. Unified Multilingual Toxicity Dataset
The primary model (**XLM-RoBERTa Multilingual Toxic**) is trained on a massive consolidation of datasets:
- **Jigsaw Multilingual Toxic Comment Classification**: Contains millions of comments from Wikipedia talk pages across 10+ languages (English, Hindi, French, Spanish, etc.), specifically labeled for toxicity, threats, and insults.
- **Civil Comments (Jigsaw 2)**: A dataset of ~2 million public comments with fine-grained toxicity labels (identity attack, insult, threat, etc.).
- **Wikipedia Detox**: Research data focused on personal attacks in academic and collaborative environments.

## 2. Speech-Specific Datasets
The **Whisper ASR** model (Audio Analysis) is trained on:
- **680,000 hours of multilingual web-audio**: Ensures high accuracy in transcribing spoken informal speech, which often differs significantly from written comments.
- **Common Voice**: A crowdsourced multilingual speech dataset.

## 3. Translation Benchmarks (Fallback)
When cross-referencing nuances, the fallback translation model uses:
- **OPUS**: A large collection of translated texts from the web, covering 100+ languages including Hindi-English pairs.
- **Helsinki-NLP benchmarks**: Focused on preserving sentiment during translation.

## 4. Fine-Tuning Focus
The models have been selected and configured to recognize:
- **Multilingual Nuance**: Detecting hate speech even when written in Romanized Hindi (Hinglish) or other language blends.
- **Implicit Toxicity**: Moving beyond simple keyword matching to understanding context.
