import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/workspace/auto_biaozhu')
import click
from loguru import logger
from tqdm import tqdm
from fish_audio_preprocess.utils.file import AUDIO_EXTENSIONS, list_files, make_dirs
import subprocess
import soundfile as sf

@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(exists=False, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Search recursively")
@click.option(
    "--overwrite/--no-overwrite", default=True, help="Overwrite existing files"
)
@click.option(
    "--clean/--no-clean", default=False, help="Clean output directory before processing"
)
@click.option(
    "--num-workers",
    help="Number of workers to use for processing, defaults to number of CPU cores",
    default=os.cpu_count(),
    show_default=True,
    type=int,
)
@click.option(
    "--min-duration",
    help="Minimum duration of each slice",
    default=5.0,
    show_default=True,
    type=float,
)
@click.option(
    "--max-duration",
    help="Maximum duration of each slice",
    default=30.0,
    show_default=True,
    type=float,
)
@click.option(
    "--min-silence-duration",
    help="Minimum duration of each slice",
    default=0.5,
    show_default=True,
    type=float,
)
@click.option(
    "--top-db",
    help="top_db of librosa.effects.split",
    default=-40,
    show_default=True,
    type=int,
)
@click.option(
    "--hop-length",
    help="hop_length of librosa.effects.split",
    default=10,
    show_default=True,
    type=int,
)
@click.option(
    "--max-silence-kept",
    help="Maximum duration of each slice",
    default=0.5,
    show_default=True,
    type=float,
)
def slice_audio_v2(
    input_dir: str,
    output_dir: str,
    recursive: bool,
    overwrite: bool,
    clean: bool,
    num_workers: int,
    min_duration: float,
    max_duration: float,
    min_silence_duration: float,
    top_db: int,
    hop_length: int,
    max_silence_kept: float,
):

    from fish_audio_preprocess.utils.slice_audio_v2 import slice_audio_file_v2

    input_dir, output_dir = Path(input_dir), Path(output_dir)

    if input_dir == output_dir and clean:
        logger.error("您正在尝试清理输入目录，正在中止")
        return

    # Convert MP3 files to 44100Hz, mono, 16-bit depth WAV
    mp3_files = list_files(input_dir, extensions=[".mp3"], recursive=recursive)
    for mp3_file in mp3_files:
        wav_file = input_dir / (mp3_file.stem + ".wav")
        command = f"ffmpeg -y -i '{mp3_file}' -ar 44100 -ac 1 -bits_per_raw_sample 16 '{wav_file}'"
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Resample existing WAV files to 44100Hz, mono, 16-bit depth
    wav_files = list_files(input_dir, extensions=[".wav"], recursive=recursive)
    for wav_file in wav_files:
        command = f"ffmpeg -y -i '{wav_file}' -ar 44100 -ac 1 -bits_per_raw_sample 16 '{wav_file}'"
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Remove MP3 files
    for mp3_file in mp3_files:
        os.remove(mp3_file)
    make_dirs(output_dir, clean)
    files = list_files(input_dir, extensions=[".wav"], recursive=recursive)
    logger.info(f"Found {len(files)} files, processing...")
    skipped = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []
        file_count = 0
        for file in tqdm(files, desc="Preparing tasks"):
            relative_path = file.relative_to(input_dir)

            output_filename_prefix = f"{output_dir.name}_{file.stem}"

            if Path(f"{output_filename_prefix}_{file_count:03d}.wav").exists() and not overwrite:
                skipped += 1
                continue

            tasks.append(
                executor.submit(
                    slice_audio_file_v2,
                    input_file=str(file),
                    output_dir=output_dir,
                    output_filename_prefix=output_filename_prefix,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    min_silence_duration=min_silence_duration,
                    top_db=top_db,
                    hop_length=hop_length,
                    max_silence_kept=max_silence_kept,
                )
            )
            file_count += 1
        for i in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            assert i.exception() is None, i.exception()

    logger.info("已完成分割操作!")

    # Calculate statistics for output files
    output_files = list_files(output_dir, extensions=[".wav"], recursive=False)
    total_duration = 0
    max_duration_file = None
    min_duration_file = None

    for file in output_files:
        try:
            sound_info = sf.info(file)
            duration = sound_info.duration
        except RuntimeError:
            logger.warning(f"无法读取文件 {file} 的时长信息，将跳过该文件统计.")
            continue

        total_duration += duration

        if max_duration_file is None or duration > max_duration_file[1]:
            max_duration_file = (file, duration)

        if min_duration_file is None or duration < min_duration_file[1]:
            min_duration_file = (file, duration)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"总共分割出: {len(output_files)}个音频, 跳过: {skipped}个音频")
    logger.info(f"最长的音频文件: {max_duration_file[0]}, 时长: {max_duration_file[1]}秒")
    logger.info(f"最短的音频文件: {min_duration_file[0]}, 时长: {min_duration_file[1]}秒")
    logger.info(f"总时长: {total_duration // 3600}小时{total_duration % 3600 // 60}分钟{total_duration % 60}秒")
    # 重命名输出文件
    file_count = 1
    for file in output_files:
        new_filename = f"{output_dir.name}_{file_count:04d}.wav"
        new_file = output_dir / new_filename
        os.rename(file, new_file)
        file_count += 1
if __name__ == "__main__":
    slice_audio_v2()