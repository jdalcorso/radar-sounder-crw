#!/bin/bash

# Define parameters
R=("45" "50" "55" "60" "65")
T=("0.1" "0.01" "0.001")
K=("15" "20" "25" "30")
# TOT: 162 runs

# Iterate through the parameter lists
for r in "${R[@]}"; do
  for t in "${T[@]}"; do
    for k in "${K[@]}"; do
        # Run Docker command with the current parameters
        docker exec -t crw_jordydalcorso python workspace/crw/scripts/test/test_all.py -r "$r" -t "$t" -k "$k"        
    done
  done
done
