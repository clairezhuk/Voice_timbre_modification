# Voice Conversion System: Jack Black Timbre Modification

This project was developed as a final assignment for the **"Acoustic Signals" (Sygnały akustyczne)** course.

## Overview
The goal of this project is **Voice Conversion (VC)**. The system takes a clean audio recording of a source voice and modifies its timbre to match a target voice while strictly preserving the original text, intonation, and melody. 

The target timbre for this project is **Jack Black** (actor and lead singer of Tenacious D). The training data was curated from open-source YouTube recordings.

## Features
- **Timbre Transfer:** High-quality voice conversion using DDSP-SVC.
- **Pitch & Content Preservation:** Uses ContentVec and Hubert to ensure the linguistic content and melody remain unchanged.
- **Custom Training Pipeline:** Scripts for preprocessing, logging, and training on custom datasets.
- **Evaluation:** Integrated metrics to evaluate the quality of the generated audio.

---

## Project Structure

The project integrates external frameworks with custom logic.

### 1. Custom Code (`/src` & root)
- `main.py`: The main entry point for the audio modification pipeline. Used to run inference by specifying input and output paths.
- `src/evaluator.py`: Handles metric evaluation for the converted audio.
- `src/feature_extractor.py`: Extracts acoustic features (F0, content embeddings) from the input audio.
- `src/synthesizer.py`: Synthesizes new audio using the trained model and the extracted features.

### 2. Training Pipeline (`/train_pipeline`)
- `main_train.py`: The entry script to start model training. Requires configuration of paths and hyperparameters in `DDSP/config.yaml`.
- `preprocess.py`: Prepares the dataset (resampling, feature extraction, alignment).
- `trainer.py`: Contains the core training loop logic.
- `logger.py`: Manages experiment logging and checkpointing.

### 3. Integrated Frameworks
This project utilizes modified versions of the following repositories:
- `DDSP_SVC_6/`: Based on [DDSP-SVC v6.1](https://github.com/yxlllc/DDSP-SVC/tree/6.1).
- `DDSP/`: Based on [DDSP-SVC Releases](https://github.com/yxlllc/DDSP-SVC/releases).
Scripts from these directories are called during the training and synthesis processes.

---

## Models and Dependencies

The system relies on several pre-trained models for feature extraction and neural vocoding:

| Component | Source / Link |
| :--- | :--- |
| **Vocoder (nsf_hifigan)** | [Nagase-Kotono/models](https://huggingface.co/Nagase-Kotono/models) |
| **Vocoder Model** | [OpenVPI Vocoders](https://github.com/openvpi/vocoders/releases) |
| **Content Extractor** | [Hubert Base (contentvec768l12.pt)](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt) |
| **Pre-trained Base** | [DDSP-SVC-Base (contentvec)](https://huggingface.co/None1145/DDSP-SVC-Base/tree/main/contentvec) |

---

## Development Process

1.  **Initial Integration:** The system was first assembled using a pre-existing third-party model ([Lappland the Decadenza](https://huggingface.co/None1145/DDSP-SVC-Lappland-the-Decadenza)) to verify the pipeline's functionality.
2.  **Dataset Preparation:** Collected and cleaned audio of Jack Black from YouTube. Audio was processed to remove background noise and music using **Ultimate Vocal Remover**.
3.  **Custom Training:** Implemented the training pipeline to train a specific model on the Jack Black dataset.
4.  **Final Implementation:** The custom-trained model was integrated into the `synthesizer.py` script for the final version of the project.

---

## Requirements

### Dependencies
All necessary Python libraries are listed in the `requirements.txt` file. To install them, run:
```bash
pip install -r requirements.txt
```

### Hardware Recommendations (GPU)
*   **Highly Recommended:** An NVIDIA GPU with CUDA support is essential for both training and inference. 
*   **Warning:** While the code can run on a CPU, the processing time will be extremely slow (especially for training and high-quality synthesis). For acceptable performance, ensure that `torch` is configured to use CUDA.

To verify if your GPU is being recognized by the system, you can run:
```python
import torch
print(torch.cuda.is_available())
```

### Software
*   **Python:** 3.8 or higher.
*   **FFmpeg:** Required for audio file processing and format conversion.
*   **CUDA Toolkit:** Ensure your version is compatible with your PyTorch installation.

---

## How to Run

### Inference (Voice Conversion)
To convert an audio file, set your input/output paths in `main.py` and run:
```bash
python main.py
```

### Training
1. Prepare your dataset in the `data/` directory.
2. Configure `DDSP/config.yaml` (set sample rate, paths, and hyperparameters).
3. Run the preprocessing script:
   ```bash
   python train_pipeline/preprocess.py
   ```
4. Start training:
   ```bash
   python train_pipeline/main_train.py
   ```
