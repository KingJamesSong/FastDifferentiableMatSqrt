#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main_cifar100.py --norm='zcanormbatch' --batch_size=128
CUDA_VISIBLE_DEVICES=1 python main_cifar10.py --norm='zcanormbatch' --batch_size=128