import getpass
import glob
import subprocess

import lightning as L
from omegaconf import OmegaConf
import os
import torch
from wandb.integration.lightning.fabric import WandbLogger
from lightning.fabric.strategies import DDPStrategy


from funcmol.utils.constants import PADDING_INDEX


def setup_fabric(config: dict, find_unused_parameters=False) -> L.Fabric:
    """
    Sets up and initializes a Lightning Fabric environment based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing the following keys:
            - "wandb" (bool): Whether to use Weights and Biases for logging.
            - "exp_name" (str): The name of the experiment.
            - "dirname" (str): The directory name for saving logs.
            - "seed" (int): The seed for random number generation.
        find_unused_parameters (bool, optional): Whether to find unused parameters in DDP strategy. Defaults to False.

    Returns:
        L.Fabric: An initialized Lightning Fabric object.
    """
    logger = None
    if config["wandb"]:
        logger = WandbLogger(
            project="funcmol",
            entity=getpass.getuser(),
            config=OmegaConf.to_container(config),
            name=config["exp_name"],
            dir=config["dirname"],
        )

    n_devs = torch.cuda.device_count()
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    strat_ = "ddp" if n_devs > 1 else "auto"
    if strat_ == "ddp" and find_unused_parameters:
        strat_ = DDPStrategy(find_unused_parameters=True)
    fabric = L.Fabric(
        devices=n_devs, num_nodes=1, strategy=strat_, accelerator="gpu", loggers=[logger], precision="bf16-mixed"
    )
    fabric.launch()
    fabric.seed_everything(config["seed"])
    fabric.print(f"config: {config}")
    fabric.print(f"device_count: {torch.cuda.device_count()}, world_size: {fabric.world_size}")

    return fabric


def mol2xyz(sample: dict, atom_elements = ["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"]):
    """
    Converts molecular data from a dictionary to an XYZ format string.

    Args:
        sample (dict): A dictionary containing molecular data with keys:
            - "atoms_channel": A tensor or array-like structure containing atom type indices.
            - "coords": A tensor or array-like structure containing atomic coordinates.
        atom_elements (list, optional): A list of element symbols corresponding to atom type indices.
                                        Defaults to ["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"].

    Returns:
        str: A string in XYZ format representing the molecular structure.
    """
    n_atoms = sample["atoms_channel"].shape[-1]
    xyz_str = str(n_atoms) + "\n\n"
    for i in range(n_atoms):
        element = sample["atoms_channel"][0, i]
        element = atom_elements[int(element.item())]
        coords = sample["coords"][0, i, :]
        element = "C" if element == "CA" else element
        line = (
            element
            + "\t"
            + str(coords[0].item())
            + "\t"
            + str(coords[1].item())
            + "\t"
            + str(coords[2].item())
        )
        xyz_str += line + "\n"
    return xyz_str


def save_xyz(mols: list, out_dir: str, fabric = None, atom_elements = ["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"]):
    def save_xyz(mols: list, out_dir: str, fabric=None, atom_elements=["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"]):
        """
        Save a list of molecules in XYZ format to the specified output directory.

        Parameters:
        mols (list): A list of molecule objects to be saved.
        out_dir (str): The directory where the XYZ files will be saved.
        fabric (optional): An object with a print method for logging. Default is None.
        atom_elements (list, optional): A list of atom elements to be considered. Default is ["C", "H", "O", "N", "F", "S", "Cl", "Br", "P", "I", "B"].

        Returns:
        list: A list of strings, each representing a molecule in XYZ format.

        Notes:
        - The function attempts to convert each molecule in the input list to XYZ format.
        - If a molecule is not valid, it is skipped, and a message is printed.
        - The number of valid molecules is logged using the fabric object's print method.
        - Each valid molecule is saved as a separate XYZ file in the output directory, with filenames in the format "sample_XXXXX.xyz".
        """
    molecules_xyz = []
    for i in range(len(mols)):
        try:
            mol = mols[i]
            xyz_str = mol2xyz(mol, atom_elements=atom_elements)
            molecules_xyz.append(xyz_str)
        except Exception:
            print(">> molecule not valid")
            continue
    fabric.print(f">> n valid molecules: {len(molecules_xyz)} / {len(mols)}")
    for idx, mol_xyz in enumerate(molecules_xyz):
        with open(os.path.join(out_dir, f"sample_{idx:05d}.xyz"), "w") as f:
            f.write(mol_xyz)
    return molecules_xyz


def atomlistToRadius(atomList: list, hashing: dict, device: str = "cpu") -> torch.Tensor:
    """
    Convert a list of atom names to their corresponding radii.

    Args:
        atomList (list): A list of atom names.
        hashing (dict): A dictionary containing the radii information for each atom.
        device (str, optional): The device to store the resulting tensor on. Defaults to "cpu".

    Returns:
        torch.Tensor: A tensor containing the radii for each atom in the input list.
    """
    radius = []
    for singleAtomList in atomList:
        haTMP = []
        for i in singleAtomList:
            resname, atName = i.split("_")[0], i.split("_")[2]
            if resname in hashing and atName in hashing[resname]:
                haTMP += [hashing[resname][atName]]
            else:
                haTMP += [1.0]
                print("missing ", resname, atName)
        radius += [torch.tensor(haTMP, dtype=torch.float, device=device)]
    radius = torch.torch.nn.utils.rnn.pad_sequence(
        radius, batch_first=True, padding_value=PADDING_INDEX
    )
    return radius


def convert_xyzs_to_sdf(path_xyzs: str, fname: str = None, delete: bool = False, fabric=None):
    fname = "molecules_obabel.sdf" if fname is None else fname
    log = (fabric.print if fabric is not None else print)
    log(f">> process .xyz files and save in .sdf in {path_xyzs}")

    # 确保路径格式正确
    path_xyzs = os.path.normpath(path_xyzs)
    output_sdf = os.path.join(path_xyzs, fname)

    # 找到 obabel
    obabel_exe = r"C:\Users\wj\.local\share\mamba\envs\fun\Library\bin\obabel.exe"
    if not os.path.exists(obabel_exe):
        log(f"obabel not found at {obabel_exe}")
        return

    # 获取所有 xyz 文件
    xyz_files = glob.glob(os.path.join(path_xyzs, "*.xyz"))
    if not xyz_files:
        log("No .xyz files found.")
        return

    # 临时输出文件
    tmp_sdf_files = []

    for i, xyz_file in enumerate(xyz_files, 1):
        sdf_tmp = os.path.join(path_xyzs, f"tmp_{i}.sdf")
        cmd = [obabel_exe, xyz_file, "-osdf", "-O", sdf_tmp, "--title", "end"]

        log(f"[{i}/{len(xyz_files)}] Converting {os.path.basename(xyz_file)} ...")
        try:
            subprocess.run(cmd, check=True)
            tmp_sdf_files.append(sdf_tmp)
        except subprocess.CalledProcessError as e:
            log(f"❌ Error converting {xyz_file}: {e}")

    # 合并所有 SDF 文件
    with open(output_sdf, "wb") as outfile:
        for sdf in tmp_sdf_files:
            with open(sdf, "rb") as infile:
                outfile.write(infile.read())

    log(f"✅ All molecules saved to: {output_sdf}")

    # 清理临时文件
    for f in tmp_sdf_files:
        try:
            os.remove(f)
        except Exception as e:
            log(f"⚠️ Failed to delete {f}: {e}")

    # 删除原始 xyz 文件（如果需要）
    if delete:
        for f in xyz_files:
            try:
                os.remove(f)
            except Exception as e:
                log(f"⚠️ Failed to delete {f}: {e}")
        log("🗑️ Deleted original .xyz files.")