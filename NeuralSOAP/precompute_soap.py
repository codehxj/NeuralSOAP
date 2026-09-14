# funcmol/precompute_soap.py
import os
import pickle
import torch
from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np
from tqdm import tqdm
import argparse
from funcmol.dataset.dataset_field import FieldDataset
from funcmol.utils.constants import ELEMENTS_HASH, PADDING_INDEX


def main():
    """可直接运行的SOAP特征预计算脚本"""
    parser = argparse.ArgumentParser(description='预计算SOAP特征')
    parser.add_argument('--dset_name', default='qm9', help='数据集名称')
    parser.add_argument('--data_dir', default='dataset/data', help='数据目录')
    parser.add_argument('--split', default='val', help='数据分割')
    parser.add_argument('--use_small', default=True, help='使用小样本数据集')  # 新增参数
    parser.add_argument('--rcut', type=float, default=6.0, help='SOAP截断半径')
    parser.add_argument('--nmax', type=int, default=8, help='SOAP nmax参数')
    parser.add_argument('--lmax', type=int, default=6, help='SOAP lmax参数')

    args = parser.parse_args()

    # 硬编码元素配置（基于QM9）
    elements = ["C", "H", "O", "N", "F"]

    # 初始化SOAP计算器
    soap = SOAP(
        species=elements,
        r_cut=args.rcut,
        n_max=args.nmax,
        l_max=args.lmax,
        periodic=False
    )

    # 加载数据集
    dataset = FieldDataset(
        dset_name=args.dset_name,
        data_dir=args.data_dir,
        elements=elements,
        split=args.split,
        rotate=False,
        data_small=args.use_small
    )

    # 创建缓存目录
    cache_dir = os.path.join(args.data_dir, args.dset_name, "soap_cache")
    os.makedirs(cache_dir, exist_ok=True)

    soap_features = []

    print(f"预计算 {args.split} 集的SOAP特征...")
    for i in tqdm(range(len(dataset.data))):
        sample = dataset.data[i]
        soap_feat = compute_soap_for_sample(sample, soap)
        soap_features.append(soap_feat)

    # 保存到磁盘 - 使用.pth格式
    # 修改缓存文件命名
    cache_suffix = "_small" if args.use_small else ""
    cache_file = os.path.join(cache_dir, f"{args.split}_soap_features{cache_suffix}.pth")

    # 将列表转换为tensor并保存
    soap_features_tensor = []
    for feat in soap_features:
        if feat is not None:
            soap_features_tensor.append(torch.tensor(feat, dtype=torch.float32))
        else:
            soap_features_tensor.append(None)

    torch.save(soap_features_tensor, cache_file)
    print(f"SOAP特征已保存到: {cache_file}")


def compute_soap_for_sample(sample, soap):
    """为单个分子样本计算SOAP特征"""
    coords = sample["coords"]
    atoms = sample["atoms_channel"]

    valid_mask = atoms != PADDING_INDEX
    if valid_mask.sum() == 0:
        return None

    valid_coords = coords[valid_mask].numpy()
    valid_atoms = atoms[valid_mask].numpy()

    atom_symbols = []
    for atom_id in valid_atoms:
        for symbol, id_val in ELEMENTS_HASH.items():
            if id_val == int(atom_id):
                atom_symbols.append(symbol)
                break

    try:
        ase_atoms = Atoms(symbols=atom_symbols, positions=valid_coords)
        soap_desc = soap.create(ase_atoms)
        return np.mean(soap_desc, axis=0)
    except Exception as e:
        print(f"计算SOAP特征时出错: {e}")
        return None


if __name__ == "__main__":
    main()