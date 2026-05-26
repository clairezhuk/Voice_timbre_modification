#!/bin/bash
rm -rf /mnt/workspace/DDSP/exp/diffusion-test/*
rm -rf /mnt/workspace/DDSP/data/train/*
rm -rf /mnt/workspace/DDSP/data/train/audio
rm -rf /mnt/workspace/DDSP/data/val/*
mkdir -p /mnt/workspace/DDSP/data/train/audio
rm -rf /mnt/workspace/DDSP/cache/*
rm -rf /mnt/workspace/DDSP/results/*
rm -rf /mnt/workspace/DDSP/raw/*

find ~ -type d -name .ipynb_checkpoints -exec rm -r {} \;

echo 不用管上面的报错，初始化完成