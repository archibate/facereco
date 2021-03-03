#!/bin/bash
set -e

build/bin/FaceLandmarkVidMulti -f /mnt/share/image.jpg
cp processed/image.csv /mnt/share/result.csv
