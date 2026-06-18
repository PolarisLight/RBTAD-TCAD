import h5py
import random
import argparse
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Configure logging for clear, informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_demo_count(filepath: Path, group_path: str) -> int:
    """Safely opens an HDF5 file and returns the number of demos."""
    try:
        with h5py.File(filepath, 'r') as f:
            return len(f[group_path])
    except Exception as e:
        logging.warning(f"Could not read demo count from {filepath.name}: {e}")
        return 0


def find_gripper_closing_timestep(gripper_states: np.ndarray, threshold: float) -> int:
    """Finds the timestep just before the gripper closes."""
    if gripper_states.shape[0] < 2:
        return 0
    diff_gripper_states = -1 * np.diff(gripper_states, axis=1)
    initial_change = diff_gripper_states[0]
    closing_candidates = np.where(np.abs(diff_gripper_states) < initial_change * threshold)[0]

    if len(closing_candidates) > 0:
        return closing_candidates[0]
    else:
        logging.warning("No clear gripper closing point found. Using fallback.")
        diff_time_gripper_states = np.diff(diff_gripper_states, axis=0)
        return np.argmin(diff_time_gripper_states)


def process_hdf5_file(
    source_path: Path, 
    dest_path: Path, 
    group_path: str, 
    threshold: float,
    back_off_steps: int
):
    """
    Reads a source HDF5 file, truncates ALL its demos, and writes to a new file.
    """
    try:
        with h5py.File(source_path, 'r') as source_f, h5py.File(dest_path, 'w') as dest_f:
            source_data = source_f.get(group_path)
            if not source_data:
                logging.warning(f"Group '{group_path}' not in {source_path.name}. Skipping.")
                return

            dest_data = dest_f.create_group(group_path)
            demo_ids = list(source_data.keys()) # Process all demos

            for demo_id in demo_ids:
                source_demo = source_data[demo_id]
                gripper_states = source_demo['obs']['gripper_states'][:]
                
                closing_step = find_gripper_closing_timestep(gripper_states, threshold)
                end_step = max(0, closing_step - back_off_steps)

                if end_step > 0:
                    dest_demo = dest_data.create_group(demo_id)
                    # Copy all datasets with the truncated length
                    for key, source_dataset_group in source_demo.items():
                        if isinstance(source_dataset_group, h5py.Group): # Handle nested 'obs'
                            dest_dataset_group = dest_demo.create_group(key)
                            for sub_key, dataset in source_dataset_group.items():
                                dest_dataset_group.create_dataset(sub_key, data=dataset[:end_step])
                        else: # Handle top-level 'actions'
                             dest_demo.create_dataset(key, data=source_dataset_group[:end_step])

    except Exception as e:
        logging.error(f"Failed to process {source_path.name}: {e}")


def run_processing(
    source_dir: Path, 
    dest_dir: Path, 
    group_path: str, 
    gripper_threshold: float,
    back_off_steps: int,
    head_task_number: int
):
    """
    Scans, ranks, and processes the top N HDF5 files based on demo count.
    """
    logging.info(f"--- Starting HDF5 dataset processing for top {head_task_number} tasks ---")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Scan all files and get their demo counts
    logging.info("Scanning all files to determine ranking...")
    all_files = list(source_dir.glob("*.hdf5"))
    if not all_files:
        logging.error(f"No .hdf5 files found in {source_dir}. Exiting.")
        return

    file_info_list = []
    for f_path in tqdm(all_files, desc="Scanning files"):
        count = get_demo_count(f_path, group_path)
        if count > 0:
            file_info_list.append({'path': f_path, 'count': count})

    # Phase 2: Sort by demo count and select the top N tasks
    file_info_list.sort(key=lambda x: x['count'], reverse=True)
    files_to_process = file_info_list[:head_task_number]
    
    logging.info(f"Identified top {len(files_to_process)} tasks to process:")
    for i, info in enumerate(files_to_process):
        logging.info(f"  Rank {i+1}: {info['path'].name}")

    # Phase 3: Process the selected files
    for file_info in tqdm(files_to_process, desc="Processing top tasks"):
        dest_path = dest_dir / file_info['path'].name
        process_hdf5_file(
            source_path=file_info['path'],
            dest_path=dest_path,
            group_path=group_path,
            threshold=gripper_threshold,
            back_off_steps=back_off_steps
        )

    logging.info(f"--- All {len(files_to_process)} selected files processed successfully! ---")


def main():
    """Parses command-line arguments and launches the processing workflow."""
    parser = argparse.ArgumentParser(
        description="Extract the target-approaching phase from the top N LIBERO HDF5 demo files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source_dir', default='dataset_all/libero_core_lt_no_noops',type=Path, help="Directory containing the source HDF5 files.")
    parser.add_argument('--dest_dir', default= 'dataset_all/libero_core_lt_no_noops_target_approaching_phase', type=Path, help="Directory to save the processed files.")
    parser.add_argument('--head_task_number', type=int, default=3, help="Number of top tasks (by demo count) to process.")
    parser.add_argument('--group_path', type=str, default='data', help="Group name inside HDF5 files containing the demos.")
    parser.add_argument('--gripper_threshold', type=float, default=0.7, help="Sensitivity threshold for detecting gripper closing event.")
    parser.add_argument('--back_off_steps', type=int, default=10, help="How many steps to back off from the closing event to define the trajectory end.")
    
    args = parser.parse_args()

    run_processing(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        group_path=args.group_path,
        gripper_threshold=args.gripper_threshold,
        back_off_steps=args.back_off_steps,
        head_task_number=args.head_task_number
    )

if __name__ == "__main__":
    main()