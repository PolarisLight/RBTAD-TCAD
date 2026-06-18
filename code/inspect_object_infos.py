import glob
import os

import h5py


files = sorted(glob.glob("dataset_all/libero_core_lt_no_noops/*.hdf5"))[:3]
print("num_files", len(files))
for file_path in files:
    print("FILE", os.path.basename(file_path))
    with h5py.File(file_path, "r") as h5:
        demo_key = sorted(h5["data"].keys())[0]
        demo = h5["data"][demo_key]
        print("demo", demo_key)
        print("keys", list(demo.keys()))
        print("obs keys", list(demo["obs"].keys()))
        if "object_infos" in demo:
            object_infos = demo["object_infos"]
            print("object_infos", object_infos.shape, object_infos.dtype)
            for idx in [0, min(5, len(object_infos) - 1), min(20, len(object_infos) - 1)]:
                value = object_infos[idx]
                print("oi", idx, value[:500] if hasattr(value, "__getitem__") else value)
        print("actions", demo["actions"].shape)
        print("ee_states", demo["obs"]["ee_states"].shape)
        print("gripper", demo["obs"]["gripper_states"].shape)
