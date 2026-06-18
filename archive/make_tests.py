import librosa
import soundfile as sf
import numpy as np
import os
import scipy.signal

def create_augmentations(name, input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    y, sr = librosa.load(input_path, sr=44100)

    # 1. Perfect match (Baseline)
    sf.write(f"{output_dir}/{name}_0_baseline.wav", y, sr)

    # 2. Pitch Shift (+12 semitones) - Imitates good clone with different voice
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=12)
    sf.write(f"{output_dir}/{name}_1_pitch_shifted.wav", y_pitch, sr)

    # 3. Additive White Noise - Imitates bad diffusion
    noise = np.random.normal(0, 0.05, len(y))
    y_noise = np.clip(y + noise, -1.0, 1.0)
    sf.write(f"{output_dir}/{name}_2_noisy.wav", y_noise, sr)

    # 4. Mechanical Chopping - Imitates F0 artifacts (blinking)
    y_choppy = np.copy(y)
    chunk_size = sr // 10 
    for i in range(0, len(y_choppy), chunk_size * 2):
        y_choppy[i:i+chunk_size] = 0.0
    sf.write(f"{output_dir}/{name}_3_choppy.wav", y_choppy, sr)

    # 5. Low-pass filter (muffled)
    b, a = scipy.signal.butter(N=4, Wn=800, fs=sr, btype='low')
    y_muffled = scipy.signal.filtfilt(b, a, y)
    sf.write(f"{output_dir}/{name}_4_muffled.wav", y_muffled, sr)

if __name__ == "__main__":
    create_augmentations("speaking",
        "voice_clone_project/data/input/speaking_ai_clean.wav", 
        "voice_clone_project/data/tests_synthetic_1"
    )
    create_augmentations("singing",
        "voice_clone_project/data/input/singing_ai_clean.wav", 
        "voice_clone_project/data/tests_synthetic_1"
    )