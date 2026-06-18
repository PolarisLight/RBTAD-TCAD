import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("NO_GCE_CHECK", "true")

import sys
sys.path.append("/mnt/data/cyh/VLA-long-tail")

from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType


data_root = Path("/mnt/data/cyh/tensorflow_datasets")
mixture_spec = [("libero_core_lt", 1.0)]
dataset_kwargs_list, sample_weights = get_oxe_dataset_kwargs_and_weights(
    data_root,
    mixture_spec,
    load_camera_views=("primary",),
    load_depth=False,
    load_proprio=False,
    load_language=True,
    action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
)
dataset, _, _ = make_interleaved_dataset(
    dataset_kwargs_list,
    sample_weights,
    train=True,
    shuffle_buffer_size=16,
    traj_transform_kwargs=dict(
        window_size=1,
        future_action_window_size=0,
        skip_unlabeled=True,
    ),
    frame_transform_kwargs=dict(
        resize_size={"primary": (224, 224)},
        num_parallel_calls=1,
    ),
    traj_transform_threads=len(mixture_spec),
    traj_read_threads=len(mixture_spec),
)

for i, batch in enumerate(dataset.as_numpy_iterator()):
    action = batch["action"]
    lang = batch["task"]["language_instruction"].decode().lower()
    print(i, "shape", action.shape, "last_values", action.reshape(-1, action.shape[-1])[-5:, -1].tolist(), "lang", lang)
    if i >= 9:
        break
