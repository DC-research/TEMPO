#!/bin/bash

# Create a new conda environment named llmts with Python 3.8
conda create -n tempo python=3.8

# Activate the llmts environment
conda activate tempo

# Install the requirements using pip
pip install -r requirements.txt