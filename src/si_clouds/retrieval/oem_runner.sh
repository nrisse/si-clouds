#!/bin/bash
# provide a file with a list of flight IDs and another file with a list of times to simulate

VERSION=$1
SETTING_FILE=$2

start_idx=$4  # Start line index (1-based)
end_idx=$5    # End line index (inclusive)

FLIGHT_IDS=()
OBS_TIMES=()

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
done < "$3"

echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] \
    VERSION: $VERSION, \
    SETTING FILE: $SETTING_FILE, \
    FLIGHT IDS: ${FLIGHT_IDS[@]}, \
    OBSERVATION TIMES: ${OBS_TIMES[@]}"

for i in "${!FLIGHT_IDS[@]}"; do
    FLIGHT_ID="${FLIGHT_IDS[$i]}"
    OBS_TIME="${OBS_TIMES[$i]}"

    echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] Retrieving: $FLIGHT_ID $OBS_TIME"
    python -m si_clouds.retrieval.oem \
        --setting_file "$SETTING_FILE" \
        --flight_id "$FLIGHT_ID" \
        --version "$VERSION" \
        --obs_time "$OBS_TIME"
done
