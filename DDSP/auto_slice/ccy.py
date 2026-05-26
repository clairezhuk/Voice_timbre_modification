import os
import argparse
import librosa
from multiprocessing import Pool, cpu_count

import soundfile
from tqdm import tqdm


def process(item):
    dirpath, filename, args = item
    wav_path = os.path.join(dirpath, filename)
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):
        wav, sr = librosa.load(wav_path, sr=args.sr)
        out_path = os.path.join(dirpath, filename)
        soundfile.write(out_path, wav, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="sampling rate",
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default="/mnt/workspace/Bert-VITS2/Data/train/audios/",
        help="path to source dir",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="cpu_processes",
    )
    args, _ = parser.parse_known_args()
    # autodl 无卡模式会识别出46个cpu
    if args.processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes = args.processes
    pool = Pool(processes=processes)

    tasks = []

    for dirpath, _, filenames in os.walk(args.in_dir):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                twople = (dirpath, filename, args)
                tasks.append(twople)

    for _ in tqdm(
        pool.imap_unordered(process, tasks),
    ):
        pass

    pool.close()
    pool.join()

    print("音频重采样完毕!")
