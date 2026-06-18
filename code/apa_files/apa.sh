# !/bin/bash
# This script performs APA (Action Phase Augmentation) on the Libero dataset.

# Step 1: Head Task Trajectory Segmentation.
python scripts/APA/segmentation.py --source_dir dataset_all/libero_core_lt_no_noops

# Step 2: Tail to Head Object Grafting.
python -m scripts.APA.grafting --source_dir dataset_all/libero_core_lt_no_noops_target_approaching_phase

# Step 3: Instruction Formatting.
python scripts/APA/formatting.py \
  --source-dir dataset_all/libero_core_lt_no_noops \
  --grafted-dir dataset_all/libero_core_lt_no_noops_target_approaching_phase_grafting \
  --target-dir dataset_all/libero_core_lt_no_noops_target_apa

# Visualization
python scripts/visualization.py --source-dir dataset_all/libero_core_lt_no_noops_target_apa

# Cleanup intermediate directories
rm -rf dataset_all/libero_core_lt_no_noops_target_approaching_phase
rm -rf dataset_all/libero_core_lt_no_noops_target_approaching_phase_grafting