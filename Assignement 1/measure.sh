#!/bin/bash
# Run the program with each file in the instances folder
for file in instances/*
do
    # Run the program with the file and time it
    echo $file
    python3 tower_sorting.py $file
    echo
done