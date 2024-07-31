#!/bin/bash

# Define parameters
S=("4" "8" "16")
L=("0.01" "0.001" "0.0001")
T=("0.1" "0.01" "0.001")
O=("16 0" "8 0" "24 0" "30 0" "0 0")
# TOT: 135 runs

# Iterate through the parameter lists
for s in "${S[@]}"; do
  for l in "${L[@]}"; do
    for t in "${T[@]}"; do
      for o in "${O[@]}"; do

        # Split the string into individual values
        overlap1=$(echo "$o" | cut -d' ' -f1)
        overlap2=$(echo "$o" | cut -d' ' -f2)

        # Run Docker command with the current parameters
        docker exec -t crw_jordydalcorso python workspace/crw/scripts/train.py --seq_length "$s" --lr "$l" --tau "$t" --overlap "$overlap1" "$overlap2"
        
      done
    done
  done
done
