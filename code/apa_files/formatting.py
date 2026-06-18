import argparse
import re
import shutil
from pathlib import Path
from typing import Callable, Optional, Tuple

# --- Constants for Regular Expressions ---
# Pre-compile regex for efficiency. Patterns are now more generic to be reusable.
PICK_UP_PATTERN = re.compile(r"^pick_up_the_(.*)_(and_place_it_.*)$")
PUSH_PUT_PATTERN = re.compile(r"^(push|put)_the_(.*?)_(to_.*|on_.*|in_.*)$")


def _extract_object_and_action(instruction: str) -> Optional[Tuple[str, str, str]]:
    """
    A helper function to parse an instruction string and extract its components.

    Args:
        instruction (str): The instruction part of a filename (without suffix).

    Returns:
        A tuple of (action, object, tail) if a match is found, otherwise None.
        e.g., ('pick_up', 'black_bowl', 'and_place_it_on_the_plate')
    """
    # Try to match the 'pick_up' pattern
    match = PICK_UP_PATTERN.match(instruction)
    if match:
        obj_raw, tail_raw = match.groups()
        return "pick_up", obj_raw, tail_raw

    # If not, try to match the 'push' or 'put' pattern
    match = PUSH_PUT_PATTERN.match(instruction)
    if match:
        action, obj_raw, tail_raw = match.groups()
        return action, obj_raw, tail_raw

    # Return None if no pattern matches
    return None


def generate_new_filename_source(filename: str) -> str:
    """
    Converts a source filename to the 'approach_then_instruct' format.
    Example: 'pick_up_the_cup_...hdf5' -> 'approach_the_cup_then_pick_it_...hdf5'
    """
    instruction = Path(filename).stem  # Remove the '.hdf5' suffix
    extracted = _extract_object_and_action(instruction)

    if not extracted:
        return f"ERROR_cannot_match_source_format_{filename}"

    action, obj_raw, tail_raw = extracted

    # Reconstruct the second part of the instruction
    if action == 'pick_up':
        instruction_tail = "pick_" + tail_raw
    else: # for 'push' or 'put'
        instruction_tail = f"{action}_it_{tail_raw}"

    return f"approach_the_{obj_raw}_then_{instruction_tail}.hdf5"


def generate_new_filename_grafted(filename: str) -> str:
    """
    Converts a grafted filename to the 'approach' only format.
    Example: 'pick_up_the_cup_..._from_...hdf5' -> 'approach_the_cup_demo.hdf5'
    """
    if '_from_' not in filename:
        return f"ERROR_format_missing_from_{filename}"

    main_instruction = filename.split('_from_')[0]
    extracted = _extract_object_and_action(main_instruction)

    if not extracted:
        return f"ERROR_cannot_match_grafted_format_{filename}"

    _, obj_raw, _ = extracted  # We only need the object for this format

    return f"approach_the_{obj_raw}_demo.hdf5"


def process_directory(
    src_dir: Path,
    tgt_dir: Path,
    rename_func: Callable[[str], str],
    start_index: int
) -> int:
    """
    Lists files in a source directory, renames them using a given function,
    prefixes them with a formatted number, and copies them to a target directory.

    Args:
        src_dir (Path): The source directory to read files from.
        tgt_dir (Path): The target directory to write files to.
        rename_func (Callable): The function used to generate the new filename.
        start_index (int): The starting number for the filename prefix.

    Returns:
        int: The next available index after processing all files.
    """
    if not src_dir.is_dir():
        print(f"Warning: Source directory not found, skipping: {src_dir}")
        return start_index

    # Get a sorted list of HDF5 files to ensure consistent processing order
    files_to_process = sorted([f.name for f in src_dir.glob('*.hdf5')])
    
    current_index = start_index
    for filename in files_to_process:
        new_name = rename_func(filename)
        # Add a sequential prefix for ordering
        prefixed_name = f"{current_index:03d}_formatting_{new_name}"

        src_path = src_dir / filename
        tgt_path = tgt_dir / prefixed_name

        shutil.copy(src_path, tgt_path)
        print(f"Copied: {filename} -> {prefixed_name}")
        current_index += 1
    
    return current_index


def main():
    """Main function to parse arguments and run the file processing."""
    parser = argparse.ArgumentParser(
        description="Rename and consolidate HDF5 trajectory files from source and grafted directories into a target APA directory."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the original source trajectory files."
    )
    parser.add_argument(
        "--grafted-dir",
        type=Path,
        required=True,
        help="Directory containing the grafted trajectory files."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory where the renamed files will be saved."
    )
    args = parser.parse_args()

    # Create the target directory if it doesn't exist
    args.target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Processing source directory: {args.source_dir} ---")
    next_index = process_directory(
        src_dir=args.source_dir,
        tgt_dir=args.target_dir,
        rename_func=generate_new_filename_source,
        start_index=0
    )
    
    print(f"\n--- Processing grafted directory: {args.grafted_dir} ---")
    process_directory(
        src_dir=args.grafted_dir,
        tgt_dir=args.target_dir,
        rename_func=generate_new_filename_grafted,
        start_index=next_index
    )
    
    print(f"\n✅ Processing complete. Renamed files are in: {args.target_dir}")


if __name__ == "__main__":
    main()