#!/usr/bin/env bash

# Record the start time (epoch seconds)
start_time=$(date +%s)

# Validate that exactly one argument is passed
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {det|llm}"
    exit 1
fi

mode="$1"

det_list=(
    "uv run agentic_system.py --source real --node det --label tn --test -1"
    "uv run agentic_system.py --source real --node det --label fp --test -1"
    "uv run agentic_system.py --source real --node det --label fn --test -1"
    "uv run agentic_system.py --source real --node det --label tp --test -1"
    "uv run agentic_system.py --source gen --node det --label tn --test -1"
    "uv run agentic_system.py --source gen --node det --label fp --test -1"
    "uv run agentic_system.py --source gen --node det --label fn --test -1"
    "uv run agentic_system.py --source gen --node det --label tp --test -1"
)

llm_list=(
    "uv run agentic_system.py --source real --node llm --label tn --test -1"
    "uv run agentic_system.py --source real --node llm --label tp --test -1"
    "uv run agentic_system.py --source gen --node llm --label tn --test -1"
    "uv run agentic_system.py --source gen --node llm --label tp --test -1"
)

case "$mode" in
    det)
        commands=("${det_list[@]}")
        ;;
    llm)
        commands=("${llm_list[@]}")
        ;;
    *)
        echo "Error: Invalid argument '$mode'. Must be 'det' or 'llm'."
        exit 1
        ;;
esac

# Execute commands from the selected list
for cmd in "${commands[@]}"; do
    echo "---------------------------------------------------------------------------------"
    echo "---------------------------------------------------------------------------------"
    echo "Running: $cmd"
    eval "$cmd"
done

# Record the end time and calculate difference
end_time=$(date +%s)
elapsed=$((end_time - start_time))

# Convert to minutes and seconds
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo "Script finished in ${minutes}m ${seconds}s (Total: ${elapsed} seconds)."

#########################
#1. in terminal run script ./data/x_run_all_test_log.txt, this will capture anything on terminal
#2. run ./run_all_test.sh det
#3. run ./run_all_test.sh llm
#4. There are color related characters in ./data/x_run_all_test_log.txt to remove
# run ansifilter ./data/x_run_all_test_log.txt > ./data/x_run_all_test_log_cleaned.txt