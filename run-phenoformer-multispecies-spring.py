
from pathlib import Path
from subprocess import call

import configs.PROBLEM_CONFIG as cfg
from configs.RUN_CONFIGS import (datasplit_configs, model_configs,
                                 training_configs)

"""
This script can run the 10 folds of each multi-task configuration.
"""

data_folder = "data/PhenoFormer-data/learning-models-data"
save_dir = "output/"
wandb_online = True
save_test_results=True

# ----------------------------------- IMPORTANT CONFIGS -----------------------------------------------------
dict_model_to_do_list = {
    "structured_temporal": ["MIRANDA"],
    "hotyear_temporal": ["MIRANDA"], 
    "highelevation_spatial": ["MIRANDA"], 
}
# --------------------------------------------------------------------------------------------------------------

full_eval = False # do evaluation for train & val as well
training_config = "default"
target = (
    "LU+NE"  # predict all leaf unfolding / needle emergence phases at the same time
)


to_do_list = ["structured_temporal", "hotyear_temporal", "highelevation_spatial"]

for split_config in to_do_list:
    for model_config in dict_model_to_do_list[split_config]:
        cmd = ["python", str(Path(__file__).resolve().parent / "cross_val_train.py")]
        unique_id = f"MultiTask-{model_config}-{target.replace(':','')}-{split_config}-{training_config}"

        run_config = {
            **model_configs[model_config],
            **training_configs[training_config],
            **datasplit_configs[split_config],
            'wandb_online': wandb_online,
            'full_eval': full_eval,
            'unique_id': unique_id,
            'save_test_results': save_test_results
        }

        for k, v in run_config.items():
            if isinstance(v, bool):
                if v:
                    cmd.extend([f"--{k}"])
            elif v is not None:
                cmd.extend([f"--{k}", str(v)])

        cmd.extend(["--data_folder", data_folder])
        cmd.extend(["--save_dir", save_dir])
        cmd.extend(["--target", target])
        cmd.extend(["--model_tag", model_config])
        cmd.extend(["--config_tag", training_config])
        cmd.extend(["--task_tag", "multispecies"])
        cmd.extend(["--cross_val_id", unique_id])
        call(cmd)