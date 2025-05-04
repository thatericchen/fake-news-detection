#!/bin/bash
# This script sets the KaggleHub cache directory to the current directory's data folder

export KAGGLEHUB_CACHE=$(pwd)/data
echo "Set KAGGLEHUB_CACHE to $KAGGLEHUB_CACHE"