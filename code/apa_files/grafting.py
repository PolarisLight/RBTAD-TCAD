# -*- coding: utf-8 -*-
"""
This script modifies HDF5 datasets for the LIBERO robotics benchmark.

It processes demonstration files from a source folder, re-renders the visual
observations using new BDDL environment configurations, and saves the modified
trajectories to an output folder. The primary use case is to generate new
datasets by replacing objects while preserving
the original robot actions.

Key functionalities:
- Reads HDF5 trajectory files from a source directory.
- For specified "tail" tasks, it generates new data by re-rendering trajectories
  from "head" tasks.
- Allows for object replacement or scene modification via new BDDL files.
- Modifies the initial state of the simulation for specific tasks.
- Saves newly rendered RGB observations and sensor data into new HDF5 files.
- Generates GIFs of the new trajectories for visual inspection.
"""

import os
import yaml
import argparse
import h5py
import numpy as np
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
print (os.getcwd())
sys.path.append('../../') 

# Set MuJoCo renderer to osmesa for offscreen rendering
os.environ["MUJOCO_GL"] = "osmesa"

import robosuite.utils.transform_utils as T
from libero.libero import benchmark
from libero.libero.envs import SegmentationRenderEnv

# Modify path to import local utilities
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
)

# Define a constant for image resolution
IMAGE_RESOLUTION = 256


class Grafting:
    """
    A class to modify and re-render LIBERO HDF5 trajectory files.

    This class handles the loading of an existing HDF5 file, setting up a new
    simulation environment based on a provided BDDL file, re-rendering the
    trajectory with potentially modified initial states, and saving the new
    dataset.
    """

    def __init__(
        self,
        source_file: str,
        output_file: str,
        task_name: str,
        source_task_name: str,
        num_demos_to_add: int,
        bddl_directory: str,
        bddl_file_path: Optional[str] = None,
    ):
        """
        Initializes the Grafting.

        Args:
            source_file (str): Path to the source HDF5 file.
            output_file (str): Path to save the modified HDF5 file.
            task_name (str): The name of the target task being generated.
            source_task_name (str): The name of the source task being used as a template.
            num_demos_to_add (int): The number of demos to select and process from the source file.
            bddl_file_path (Optional[str]): Path to the new BDDL file for environment setup.
        """
        self.source_file = Path(source_file)
        self.output_file = Path(output_file)
        self.task_name = task_name
        self.source_task_name = source_task_name
        self.num_demos_to_add = num_demos_to_add
        self.bddl_file_path = bddl_file_path
        self.temp_dir = tempfile.mkdtemp()
        try:
            state_mods_path = os.path.join(bddl_directory, 'state_mods.yaml')
            with open(state_mods_path, 'r') as f:
                self.state_mods_config = yaml.safe_load(f)
        except FileNotFoundError:
            print("Warning: state_mods.yaml not found. No state modifications will be applied.")
            self.state_mods_config = {}

    def modify_hdf5(self) -> None:
        """
        Main method to perform the HDF5 file modification.

        Orchestrates the process of reading, re-rendering, and writing the new
        trajectory data.
        """
        if not self.source_file.exists():
            print(f"Warning: Source file not found, skipping: {self.source_file}")
            return
            
        if self.num_demos_to_add == 0:
            print(f"Info: No demos to add for {self.task_name} from {self.source_task_name}. Skipping.")
            return

        print(f"Task: {self.task_name}")
        print(f"Processing source file: {self.source_file}")

        try:
            # Create the modified task configuration for the simulation environment
            modified_task = self._create_modified_task()

            with h5py.File(self.source_file, "r") as orig_file:
                orig_data = orig_file["data"]
                
                with h5py.File(self.output_file, "w") as new_file:
                    new_data_grp = new_file.create_group("data")

                    # Select a random subset of demos to process
                    key_list = list(orig_data.keys())
                    selected_keys = np.random.choice(key_list, size=self.num_demos_to_add, replace=False)
                    print(f"Selected demo keys: {selected_keys}")

                    for demo_key in selected_keys:
                        print(f"  Processing demo: {demo_key}...")
                        self._process_demo(orig_data[demo_key], new_data_grp, demo_key, modified_task)

            print(f"Successfully created modified file: {self.output_file}\n")

        finally:
            # Clean up temporary directory
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

    def _create_modified_task(self) -> Optional[object]:
        """
        Creates a mock task object with the new BDDL file path.

        This allows the simulation environment to be initialized with the
        correct (modified) scene and object configuration.
        """
        try:
            # A simple container class to mimic the structure of a LIBERO task object
            class ModifiedTask:
                def __init__(self, name, bddl_path):
                    self.name = name
                    self.bddl_file_path = bddl_path

            return ModifiedTask(self.source_task_name, self.bddl_file_path)
        except Exception as e:
            print(f"Error creating modified task configuration: {e}")
            return None

    def _setup_environment(self, modified_task: object) -> SegmentationRenderEnv:
        """
        Sets up the simulation environment for re-rendering.
        
        Args:
            modified_task (object): The mock task object with BDDL info.
            
        Returns:
            SegmentationRenderEnv: The initialized simulation environment.
        """
        env_args = {
            "bddl_file_name": modified_task.bddl_file_path,
            "camera_heights": IMAGE_RESOLUTION,
            "camera_widths": IMAGE_RESOLUTION,
        }
        env = SegmentationRenderEnv(**env_args)
        print(f"  Environment created with BDDL: {modified_task.bddl_file_path}")
        return env

    def _process_demo(
        self,
        orig_demo_data: h5py.Group,
        new_data_grp: h5py.Group,
        demo_key: str,
        modified_task: object,
    ) -> None:
        """
        Processes a single demonstration: re-renders and saves all data.

        Args:
            orig_demo_data (h5py.Group): The original demo data group.
            new_data_grp (h5py.Group): The HDF5 group to save new data into.
            demo_key (str): The key for the current demonstration (e.g., 'demo_0').
            modified_task (object): The mock task object for environment setup.
        """
        new_demo_grp = new_data_grp.create_group(demo_key)

        # Copy original trajectory data
        orig_actions = orig_demo_data["actions"][()]
        orig_states = orig_demo_data["states"][()]

        if "robot_states" in orig_demo_data:
            new_demo_grp.create_dataset("robot_states", data=orig_demo_data["robot_states"][()])
        
        # Ensure rewards and dones are present, creating defaults if necessary
        rewards = orig_demo_data["rewards"][()] if "rewards" in orig_demo_data else np.zeros(len(orig_actions), dtype=np.uint8)
        dones = orig_demo_data["dones"][()] if "dones" in orig_demo_data else np.zeros(len(orig_actions), dtype=np.uint8)
        if not (rewards.any() and dones.any()):
            rewards[-1], dones[-1] = 1, 1

        # Set up the environment and re-render the trajectory
        env = self._setup_environment(modified_task)
        if env is not None:
            new_obs_data = self._rerender_demo(env, orig_states, orig_actions)
            env.close()
        else:
            # Fallback: if environment creation fails, just copy original observations
            print("  Warning: Environment setup failed. Copying original observation data.")
            new_obs_data = {key: orig_demo_data["obs"][key][()] for key in orig_demo_data["obs"].keys()}

        # Save all new data to the HDF5 file
        new_obs_grp = new_demo_grp.create_group("obs")
        for key, data in new_obs_data.items():
            if data.size > 0:  # Only save datasets with content
                new_obs_grp.create_dataset(key, data=data)

        new_demo_grp.create_dataset("actions", data=orig_actions)
        new_demo_grp.create_dataset("states", data=orig_states)
        new_demo_grp.create_dataset("rewards", data=rewards)
        new_demo_grp.create_dataset("dones", data=dones)

        if "object_infos" in orig_demo_data:
            new_demo_grp.create_dataset("object_infos", data=orig_demo_data["object_infos"][()])

        # Save a GIF for visualization
        if "agentview_rgb" in new_obs_data and new_obs_data["agentview_rgb"].size > 0:
            self._save_trajectory_gif(new_obs_data["agentview_rgb"], demo_key)

    def _rerender_demo(
        self, env: SegmentationRenderEnv, orig_states: np.ndarray, orig_actions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Replays a trajectory in the new environment to generate new observations.

        Args:
            env (SegmentationRenderEnv): The simulation environment.
            orig_states (np.ndarray): The sequence of simulation states from the original demo.
            orig_actions (np.ndarray): The sequence of actions from the original demo.

        Returns:
            Dict[str, np.ndarray]: A dictionary of the newly rendered observation data.
        """
        env.reset()
        initial_state = self._apply_task_specific_state_mods(orig_states[0])
        env.set_init_state(initial_state)

        # Allow the simulation to settle for a few steps
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action("llava"))

        # Data structure to collect new observations
        obs_data = {
            'agentview_rgb': [], 'eye_in_hand_rgb': [], 'gripper_states': [],
            'joint_states': [], 'ee_states': [], 'ee_pos': [], 'ee_ori': []
        }

        # Collect the very first observation after reset
        self._collect_observation(obs, obs_data)

        # Replay the action sequence
        for action in orig_actions:
            obs, _, _, _ = env.step(action)
            self._collect_observation(obs, obs_data)
        
        # Convert lists of arrays into stacked numpy arrays
        for key in obs_data:
            if obs_data[key]:
                obs_data[key] = np.stack(obs_data[key], axis=0)
            else:
                obs_data[key] = np.array([])
        
        print(f"  Successfully re-rendered {len(orig_actions) + 1} timesteps.")
        return obs_data

    def _apply_task_specific_state_mods(self, state: np.ndarray) -> np.ndarray:
        """
        Applies modifications to the initial state based on a YAML config file.
        """
        # Check if the current task has a modification entry in the config
        if self.task_name in self.state_mods_config:
            mods = self.state_mods_config[self.task_name]
            
            # Apply quaternion modification if it exists
            if 'quaternion' in mods:
                state[13:17] = mods['quaternion']
                
            # Apply x-position offset if it exists
            if 'x_pos_offset' in mods:
                state[12] += mods['x_pos_offset']
                
        return state

    def _collect_observation(self, obs: Dict, obs_data: Dict[str, List]) -> None:
        """
        Extracts relevant data from a single observation dictionary.

        Args:
            obs (Dict): The observation dictionary from `env.step()`.
            obs_data (Dict[str, List]): The dictionary to append data to.
        """
        # Collect image data, flipping vertically to correct orientation
        if "agentview_image" in obs:
            img = get_libero_image(obs, IMAGE_RESOLUTION, "agentview_image")
            obs_data['agentview_rgb'].append(np.flipud(img))

        if "robot0_eye_in_hand_image" in obs:
            img = get_libero_image(obs, IMAGE_RESOLUTION, "robot0_eye_in_hand_image")
            obs_data['eye_in_hand_rgb'].append(img)

        # Collect robot state data
        if "robot0_gripper_qpos" in obs:
            obs_data['gripper_states'].append(obs["robot0_gripper_qpos"])
        if "robot0_joint_pos" in obs:
            obs_data['joint_states'].append(obs["robot0_joint_pos"])
        if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
            eef_pos = obs["robot0_eef_pos"]
            eef_ori = T.quat2axisangle(obs["robot0_eef_quat"])
            obs_data['ee_pos'].append(eef_pos)
            obs_data['ee_ori'].append(eef_ori)
            obs_data['ee_states'].append(np.hstack((eef_pos, eef_ori)))

    def _save_trajectory_gif(self, images: np.ndarray, demo_key: str) -> None:
        """
        Saves an image sequence as a GIF file.

        Args:
            images (np.ndarray): A sequence of RGB images.
            demo_key (str): The key of the demonstration, used in the filename.
        """
        try:
            from PIL import Image
        except ImportError:
            print("Warning: Pillow is not installed. Skipping GIF generation. `pip install Pillow`")
            return

        gif_dir = self.output_file.parent / "gifs" / self.task_name
        gif_dir.mkdir(parents=True, exist_ok=True)
        gif_path = gif_dir / f"{demo_key}_{self.source_task_name}.gif"

        frames = [Image.fromarray(np.flipud(img)) for img in images]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"  GIF saved to: {gif_path}")


def get_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Tail to Head Object Grafting")
    parser.add_argument("--source_dir", type=str, default="dataset_all/libero_core_lt_no_noops_target_approaching_phase", help="Folder containing source HDF5 files.")
    parser.add_argument("--output_dir", type=str, default="dataset_all/libero_core_lt_no_noops_target_approaching_phase_grafting", help="Folder to save the modified HDF5 files.")
    parser.add_argument("--bddl_dir", type=str, default="scripts/APA/bddls", help="Base folder containing BDDL files for new tasks.")
    parser.add_argument("--generate_number", type=int, default=6, help="Number of new demos to generate for each target task.")
    parser.add_argument("--head_number", type=int, default=3, help="Number of source tasks (head tasks) to use for generating new data.")
    parser.add_argument("--tail_number", type=int, default=7, help="Number of target tasks (tail tasks) to generate.")
    return parser.parse_args()


def main():
    """Main execution function."""
    args = get_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define the set of tasks to process and the number of demos to generate for each
    # Note: These lists are tightly coupled by index.
    TARGET_TASKS = [
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_bowl_on_top_of_the_cabinet",
        "put_the_cream_cheese_in_the_bowl",
        "put_the_wine_bottle_on_top_of_the_cabinet",
        "put_the_wine_bottle_on_the_rack",
    ]
    
    # Number of new demos to generate for each corresponding TARGET_TASK
    NUM_DEMOS_TO_GENERATE = [0]*args.head_number + [args.generate_number]*args.tail_number
    
    np.random.seed(99) # For reproducible demo selection

    # The first `head_number` tasks are considered sources for generating the rest
    for i, target_task_name in enumerate(TARGET_TASKS):
        if i < args.head_number:
            continue # Skip source tasks, as we are only generating data for target tasks

        num_to_add_total = NUM_DEMOS_TO_GENERATE[i]
        if num_to_add_total == 0:
            continue
            
        # Distribute the total number of demos to add evenly among the source tasks
        base_count = num_to_add_total // args.head_number
        remainder = num_to_add_total % args.head_number
        num_demos_per_source = [base_count] * args.head_number
        for k in range(remainder):
            num_demos_per_source[k] += 1

        # The BDDL folder for the target task contains different scene variations
        target_bddl_dir = Path(args.bddl_dir) / target_task_name
        if not target_bddl_dir.exists():
            print(f"Warning: BDDL folder not found for task '{target_task_name}', skipping.")
            continue
        
        # BDDL files define the environment. Each corresponds to a source task.
        # We assume they are sorted alphabetically.
        bddl_files = sorted(os.listdir(target_bddl_dir))

        for j, bddl_filename in enumerate(bddl_files):
            source_task_name = Path(bddl_filename).stem
            bddl_file_path = target_bddl_dir / bddl_filename
            
            print("-" * 50)
            print(f"Processing target task {i+1}/{len(TARGET_TASKS)}: {target_task_name}")
            print(f"Using source: {source_task_name} with BDDL: {bddl_filename}")

            modifier = Grafting(
                source_file=Path(args.source_dir) / f"{source_task_name}_demo.hdf5",
                output_file=output_path / f"{target_task_name}_from_{source_task_name}_demo.hdf5",
                task_name=target_task_name,
                source_task_name=source_task_name,
                num_demos_to_add=num_demos_per_source[j],
                bddl_directory=args.bddl_dir,
                bddl_file_path=str(bddl_file_path),
            )
            modifier.modify_hdf5()

    print("=" * 50)
    print("All tasks processed successfully!")


if __name__ == "__main__":
    main()