#!/bin/bash

python3 main.py vae -p results --hidden-size 512 --batch-size 40
python3 main.py vae -p results --hidden-size 512 --batch-size 40 --residual
python3 main.py vae -p results --hidden-size 512 --batch-size 40 --strides=dilation
python3 main.py vae -p results --hidden-size 512 --batch-size 40 --strides=dilation --residual


python3 main.py wgan -p results --hidden-size 128 --batch-size 40
python3 main.py wgan -p results --hidden-size 256 --batch-size 40
python3 main.py wgan -p results --hidden-size 512 --batch-size 40
python3 main.py wgan -p results --hidden-size 512 --batch-size 40 --residual
python3 main.py wgan -p results --hidden-size 512 --batch-size 40 --strides=dilation
python3 main.py wgan -p results --hidden-size 512 --batch-size 40 --strides=dilation --residual

