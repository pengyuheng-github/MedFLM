#!/bin/bash
module load anaconda/2020.11
source activate medgpt
export PYTHONUNBUFFERED=1
python try_prompter.py