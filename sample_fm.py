import os
import torch
import omegaconf
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata
import typing

from funcmol.models.funcmol import create_funcmol
from funcmol.train_fm import sample
from funcmol.utils.utils_base import setup_fabric
from funcmol.utils.utils_nf import load_neural_field
from funcmol.utils.utils_fm import load_checkpoint_fm

import hydra

# 支持旧 checkpoint 类型
torch.serialization.add_safe_globals([
    DictConfig,
    ContainerMetadata,
    typing.Any,
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    dict, list, tuple, set,
])

@hydra.main(config_path="configs", config_name="sample_fm", version_base=None)
def main(config):
    fabric = setup_fabric(config)

    ckpt_path = os.path.join(config["fm_pretrained_path"], "checkpoint.pth.tar")
    fabric.print(f">> Loading FuncMol checkpoint: {ckpt_path}")
    checkpoint_fm = torch.load(ckpt_path, map_location="cpu")

    # Merge config
    config_ckpt = checkpoint_fm["config"]
    for key in config.keys():
        if key in config_ckpt and isinstance(config_ckpt[key], omegaconf.dictconfig.DictConfig):
            config_ckpt[key].update(config[key])
        else:
            config_ckpt[key] = config[key]
    config = config_ckpt
    fabric.print(f"Updated config: {config}")

    # Load FuncMol
    with torch.no_grad():
        funcmol = create_funcmol(config, fabric)
        funcmol, code_stats = load_checkpoint_fm(funcmol, config["fm_pretrained_path"], fabric=fabric)
        funcmol = fabric.setup_module(funcmol)

        # Load Neural Field
        nf_ckpt_path = os.path.join(config["nf_pretrained_path"], "model.pt")
        fabric.print(f">> Loading NF checkpoint: {nf_ckpt_path}")
        nf_checkpoint = torch.load(nf_ckpt_path, map_location="cpu", weights_only=False)
        _, dec = load_neural_field(nf_checkpoint, fabric, config=nf_checkpoint["config"])
        dec_module = dec.module if hasattr(dec, "module") else dec
        dec_module.set_code_stats(code_stats)

    # ✅ Sampling 不传 n_atoms
    fabric.print(">> Saving samples in", config["dirname"])
    sample(funcmol, dec_module, config, fabric)

if __name__ == "__main__":
    main()
