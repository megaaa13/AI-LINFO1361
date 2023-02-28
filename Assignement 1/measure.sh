#!/bin/bash

MAX_TIME=3m
MEASURE_FILE=measure.txt

# Reset the measure file
echo "Starting measurement" > ${MEASURE_FILE}
echo "Starting measurement. This may take a while (max 120m)"

# Measure function
measure() {
    echo "" >> ${MEASURE_FILE}
    echo "Measuring $2 with $3" >> ${MEASURE_FILE}
    timeout -s 10 ${MAX_TIME} python3 $1 $2 >> ${MEASURE_FILE}
}

# Measure all the files
export PYTHONPATH=aima-python3/
for file in instances/* ; do
    echo "Measuring $file"
    measure tower_sorting_bfsg.py $file bfsg
    measure tower_sorting_bfst.py $file bfst
    measure tower_sorting_dfsg.py $file dfsg
    measure tower_sorting_dfst.py $file dfst
done