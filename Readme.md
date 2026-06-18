# Voice Conversion System: Jack Black Timbre Modification

This project was developed as a final assignment for the **"Acoustic Signals"** course.

## Overview

The goal of this project is **Voice Conversion**. The system takes a clean audio recording of a source voice and modifies its timbre to match a target voice while strictly preserving the original text, intonation, and melody.

The target timbre for this project is **Jack Black** (actor and lead singer of Tenacious D). The training data was curated from open-source YouTube recordings.

## Features

* **Timbre Transfer:** High-quality voice conversion using DDSP-SVC.
* **Pitch & Content Preservation:** Uses ContentVec and Hubert to ensure the linguistic content and melody remain unchanged.
* **Custom Training Pipeline:** Scripts for preprocessing, logging, and training on custom datasets.
* **Evaluation:** Integrated metrics to evaluate the quality of the generated audio.

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone [your-github-repo-link]
cd [repo-name]

```

### 2. Install Dependencies

All necessary Python libraries are listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt

```

### 3. Download Model Weights

Due to file size limitations on GitHub, the `.pt` model weights are hosted on Hugging Face:

🔗 **[ZhClr/Voice_modification_diffusion](https://huggingface.co/ZhClr/Voice_modification_diffusion)**

You can download these folders manually or using Git LFS and place them in the project root:

* **`models/` (Required):** Contains the necessary models for running inference (`main.py`). Place downloaded models here.
* **`my_model/` (Optional):** Contains weights generated during the training process. **Warning: Large size (~7.9 GB).** * *Skip this* if you only want to run inference.
* If you plan to resume training, you can download just the specific checkpoint you need.
* If left empty, a new model with random weights will be initialized.
* You can also place a different compatible model here (must match format in `DDSP/config.yaml`). `my_model/config.yaml` syncs automatically.
* *Note:* Existing logs in this Hugging Face folder contain my personal training history and can be safely deleted. New logs will generate automatically.


* **`my_model_archive_try/` (Optional):** Contains weights from experimental training runs. You can ignore or delete this.

---

## Project Structure

The project integrates external frameworks with custom logic.

### 1. Custom Code (`/src` & root)

* `main.py`: The main entry point for the audio modification pipeline. Used to run inference.
* `src/evaluator.py`: Handles metric evaluation for the converted audio.
* `src/feature_extractor.py`: Extracts acoustic features (F0, content embeddings).
* `src/synthesizer.py`: Synthesizes new audio using the trained model.

### 2. Training Pipeline (`/train_pipeline`)

* `main_train.py`: The entry script to start model training.
* `preprocess.py`: Prepares the dataset (resampling, feature extraction, alignment).
* `trainer.py`: Contains the core training loop logic.
* `logger.py`: Manages experiment logging and checkpointing.

### 3. Data & Archives

* `archive/`: Contains scripts and parameters from experimental attempts. Not required for the main pipeline.
* `data/`: Contains sliced `.wav` audio fragments used for training. *Note: Preprocessed features are not included in the commits.*

### 4. Integrated Frameworks

* `DDSP_SVC_6/`: Based on [DDSP-SVC v6.1](https://github.com/yxlllc/DDSP-SVC/tree/6.1).
* `DDSP/`: Based on [DDSP-SVC Releases](https://github.com/yxlllc/DDSP-SVC/releases).

---

## Models and Dependencies

The system relies on several pre-trained models for feature extraction and neural vocoding:

| Component | Source / Link |
| --- | --- |
| **Vocoder (nsf_hifigan)** | [Nagase-Kotono/models](https://huggingface.co/Nagase-Kotono/models) |
| **Vocoder Model** | [OpenVPI Vocoders](https://github.com/openvpi/vocoders/releases) |
| **Content Extractor** | [Hubert Base (contentvec768l12.pt)](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt) |
| **Pre-trained Base** | [DDSP-SVC-Base (contentvec)](https://huggingface.co/None1145/DDSP-SVC-Base/tree/main/contentvec) |

---

## Development Process

1. **Initial Integration:** The system was assembled using a pre-existing third-party model ([Lappland the Decadenza](https://huggingface.co/None1145/DDSP-SVC-Lappland-the-Decadenza)) to verify the pipeline.
2. **Dataset Preparation:** Collected and cleaned audio of Jack Black from YouTube using **Ultimate Vocal Remover**.
3. **Custom Training:** Implemented the training pipeline to train a specific model on the Jack Black dataset.
4. **Final Implementation:** The custom-trained model was integrated into the `synthesizer.py` script.

---

## Hardware Recommendations (GPU)

* **Highly Recommended:** An NVIDIA GPU with CUDA support is essential.
* **Warning:** Running on a CPU is possible but will be extremely slow. Ensure `torch` is configured to use CUDA:

```python
import torch
print(torch.cuda.is_available())

```

---

## How to Run

### Inference (Voice Conversion)

1. Place the audio files you want to convert in `data/dataset/test/` (or any convenient folder).
2. Open `main.py` and adjust the configuration paths:
* `INPUT_PATH`: Path to your source audio.
* `OUTPUT_PATH`: Path where the converted audio will be saved.
* `RES_PATH`: Directory to save evaluation metrics.


3. Run the script:

```bash
python main.py

```

### Training

1. **Prepare Data:** Place your new `.wav` audio files into `data/dataset/train/audio` and `data/dataset/val/audio`.
2. **Configure:** Set your paths and hyperparameters in `DDSP/config.yaml`.
3. **Preprocess:** In `train_pipeline/main_train.py`, locate the preprocessing step:
```python
print("=== STEP 1: Preprocessing ===")
preprocess_dataset(CONFIG)

```


**Important:** You only need to run this step *once* for new data. On subsequent runs, comment these lines out to skip preprocessing.
4. **Start Training:**
```bash
python train_pipeline/main_train.py

```


5. **Resume Training:** You can manually stop the training process at any time (e.g., using `Ctrl+C`). When you run `main_train.py` again, training will automatically resume from the last saved filesystem checkpoint (configured in `config.yaml`). *Remember to comment out the preprocessing step when resuming.*