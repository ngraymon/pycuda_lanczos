#!/bin/bash

## declare our arrays
declare -a basis_size_list=( 100 200 300 350 400 450 500 )
declare -a iterations_list=( 100 200 400 ) 

# get length of arrays
iter_length=${#iterations_list[@]}
basis_length=${#basis_size_list[@]}
OUTPUT_FILE_NAME=cpu_timing_output.txt

pids=""

for (( i=1; i<${iter_length}+1; i++ )); do
	for (( b=1; b<${basis_length}+1; b++ )); do
		echo "Iter length = ${iterations_list[$i-1]}, Basis size = ${basis_size_list[$b-1]}"
		echo "${iterations_list[$i-1]} Iterations, basis size: ${basis_size_list[$b-1]}" > ./logs/io003/"${basis_size_list[$b-1]}_${iterations_list[$i-1]}$OUTPUT_FILE_NAME"
		./../lanczosHO/harmonic3d ${basis_size_list[$b-1]} ${iterations_list[$i-1]} 100 -10 10 >> ./logs/io003/"${basis_size_list[$b-1]}_${iterations_list[$i-1]}$OUTPUT_FILE_NAME"  &
		pids="$pids $!"
	done
done

wait $pids
echo "All jobs complete"