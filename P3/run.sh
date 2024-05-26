#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 main.py wgan -p wgan --hidden-size 512 --batch-size 100
