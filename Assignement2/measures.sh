#!/bin/bash

MAX_TIME=300s
MEASURE_FILE=measure.txt

# Reset the measure file
echo "Starting measurement" > ${MEASURE_FILE}
echo "Starting measurement. This may take a while"

# Measure function
measure() {
    echo "" >> ${MEASURE_FILE}
    echo "Measuring $2 with $3" >> ${MEASURE_FILE}
    timeout ${MAX_TIME} python3 $1 $2 >> ${MEASURE_FILE}
}

# Measure all the files
export PYTHONPATH=aima-python3/
for file in instances/* ; do
    echo "Measuring $file"
    measure softflow.py $file
done