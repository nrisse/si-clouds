#!/bin/bash
# provide a file with a list of flight IDs and another file with a list of times to simulate

VERSION=$1
SETTING_FILE=$2

CWP_DISTRIBUTION=$4  # uniform or gaussian
start_idx=$5  # Start line index (1-based)
end_idx=$6    # End line index (inclusive)

FLIGHT_IDS=()
OBS_TIMES=()
RANDOM_SEEDS=()

current_idx=0
while IFS=, read -r obs_time flight_id; do
    ((current_idx++))
    
    if ((current_idx < start_idx)); then
        continue
    fi
    
    if ((current_idx > end_idx)); then
        break
    fi
    
    OBS_TIMES+=("$obs_time")
    FLIGHT_IDS+=("$flight_id")
    RANDOM_SEEDS+=("$current_idx")
done < "$3"

echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] \
    VERSION: $VERSION, \
    SETTING FILE: $SETTING_FILE, \
    FLIGHT ID: $FLIGHT_ID, \
    OBSERVATION TIMES: ${OBS_TIMES[@]}"

for i in "${!FLIGHT_IDS[@]}"; do
    FLIGHT_ID="${FLIGHT_IDS[$i]}"
    OBS_TIME="${OBS_TIMES[$i]}"
    RANDOM_SEED="${RANDOM_SEEDS[$i]}"

    echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] Synthetic retrieval for: $FLIGHT_ID $OBS_TIME with random seed $RANDOM_SEED and $CWP_DISTRIBUTION CWP distribution"
    python -m si_clouds.retrieval.synthetic \
        --setting_file "$SETTING_FILE" \
        --flight_id "$FLIGHT_ID" \
        --version "$VERSION" \
        --obs_time "$OBS_TIME" \
        --syn_type "random" \
        --test_id "random" \
        --cwp_distribution "$CWP_DISTRIBUTION" \
        --random_seed "$RANDOM_SEED"
done
