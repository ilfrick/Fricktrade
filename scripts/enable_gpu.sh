#!/bin/bash
# Re-enable GPU for Fricktrade training

echo '{"gpu_disabled": false}' > /home/nicola/Fricktrade/data/gpu_state.json
chown nicola:nicola /home/nicola/Fricktrade/data/gpu_state.json
echo "GPU re-enabled. Restart trainer to use GPU."
