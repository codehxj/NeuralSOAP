import math
import numpy as np
import torch
from torch import nn
from funcmol.utils.constants import PADDING_INDEX

class FieldMaker(nn.Module):
    def __init__(
        self,
        config: dict,
        cubes_around: int = 5,  #在中心点周围放置的立方体数目（默认为 5）
        sample_points: bool = True,  # 是否计算 点的占据情况。
        sample_grid: bool = True,  #是否计算 网格的占据情况
    ):
        """
        Initializes the FieldMaker class with the given configuration.

        Args:
            config (dict): Configuration dictionary containing dataset parameters.
            cubes_around (int, optional): Number of cubes around the central point. Defaults to 5.
            sample_points (bool, optional): Flag to sample points. Defaults to True.
            sample_grid (bool, optional): Flag to sample grid. Defaults to True.

        Attributes:
            grid_dim_low_res (int): Low resolution grid dimension.
            resolution_low_res (float): Low resolution grid resolution.
            grid_dim (int): Grid dimension.
            resolution (float): Grid resolution.
            radius (float): Atomic radius.
            radius_scale (float): Scaled atomic radius.
            cubes_around (int): Number of cubes around the central point.
            num_channels (int): Number of channels in the dataset.
            vol_maker (Voxels): Instance of the Voxels class.
            elements (list): List of elements in the dataset.
            sample_points (bool): Flag to sample points.
            sample_grid (bool): Flag to sample grid.
        """
        super(FieldMaker, self).__init__() #这些参数主要用于定义 网格尺寸、分辨率、原子半径、原子通道数目 等信息
        self.grid_dim_low_res = config["dset"]["grid_dim"] // config["dset"]["subsampling_ratio"]
        self.resolution_low_res = config["dset"]["resolution"] * config["dset"]["subsampling_ratio"]
        self.grid_dim = config["dset"]["grid_dim"]
        self.resolution = config["dset"]["resolution"]
        self.radius = config["dset"]["atomic_radius"]
        self.radius_scale = np.sqrt(config["dset"]["subsampling_ratio"])  #计算 原子半径的缩放比例，用于匹配低分辨率网格。
        self.cubes_around = cubes_around
        self.num_channels = config["dset"]["n_channels"]
        self.vol_maker = Voxels()  # Voxels 是一个体积计算模块，用于生成体素（volumes）。
        self.elements = config["dset"]["elements"]
        self.sample_points = sample_points
        self.sample_grid = sample_grid

    @torch.no_grad()
    def forward(self, batch: dict)-> tuple:
        # atomic occupancies of points  计算点的原子占据情况
        occs_points = None
        if self.sample_points:
            occs_points = self._compute_occupancies(batch)
            if not occs_points.is_contiguous():  # making it compatible with torch.compile
                occs_points = occs_points.contiguous()

        # atomic occupancies of grid  计算网格的原子占据情况
        occs_grid = None
        if self.sample_grid:
            # dumb coordinates to center molecule
            batch = self._add_dumb_coords(batch)  ## 生成用于对齐分子的“哑”坐标
            occs_grid = self.vol_maker(
                batch["coords"],
                batch["radius"] * self.radius_scale,
                batch["atoms_channel"],
                resolution=self.resolution_low_res,
                cubes_around_atoms_dim=self.cubes_around,
                numberchannels=self.num_channels,
            )
            # get center box (and remove dumb coordinates) # 取中心区域
            c = occs_grid.shape[-1] // 2
            box_min, box_max = c - self.grid_dim_low_res // 2, c + self.grid_dim_low_res // 2
            occs_grid = occs_grid[:, :, box_min:box_max, box_min:box_max, box_min:box_max]
            if not occs_grid.is_contiguous():  # making it compatible with torch.compile
                occs_grid = occs_grid.contiguous()

        return occs_points, occs_grid

    def _add_dumb_coords(self, batch: dict) -> dict:  #用于添加 额外的“哑”坐标，以确保分子中心对齐：
        """
        Add dumb coordinates to center the molecule.

        Args:
            batch (dict): A dictionary containing the molecular data.

        Returns:
            dict: A dictionary containing the molecular data with dumb coordinates added.
        """
        bsz = batch['coords'].shape[0]
        dumb_coord = batch['coords'][batch['coords'] != torch.tensor(PADDING_INDEX, dtype=batch['coords'].dtype)].abs().max() + 10
        dumb_coord_repeat = dumb_coord.repeat(bsz, 1, 3)
        return {
            "coords": torch.cat((batch['coords'], -dumb_coord_repeat, dumb_coord_repeat), 1),
            "atoms_channel": torch.cat(
                (batch['atoms_channel'], torch.full((bsz, 2), 0, dtype=batch['atoms_channel'].dtype, device=batch['atoms_channel'].device)), 1
            ),
            "radius": torch.cat(
                (batch['radius'], torch.full((bsz, 2), self.radius, dtype=batch['radius'].dtype, device=batch['radius'].device), ), 1
            )
        }

    def _compute_occupancies(self, mols) -> torch.Tensor:
        """
        Compute the occupancies for a given set of molecules.

        Args:
            mols (dict): A dictionary containing molecular data with the following keys:
                - "xs" (torch.Tensor): Scaled coordinates of the molecules.
                - "atoms_channel" (torch.Tensor): Channel indices for the atoms.
                - "coords" (torch.Tensor): Coordinates of the atoms.

        Returns:
            torch.Tensor: A tensor containing the computed occupancies for each molecule.

        Notes:
            - The coordinates are scaled to be in Angstrom units.
            - The occupancy is computed based on the distance between the scaled coordinates and the atom coordinates.
            - The exponent is clamped to a maximum value of 10 to avoid unnecessary computation.
            - The occupancy is computed for valid channels, excluding the padding index.
        """
        # scale the xs to be in Angstrom
        xs = mols["xs"] * (self.resolution * self.grid_dim / 2)
        occs = torch.zeros(xs.size(0), xs.size(1), len(self.elements), device=xs.device)
        unique_channels = mols["atoms_channel"].int().unique()
        valid_channels = unique_channels[unique_channels != torch.tensor(PADDING_INDEX, dtype=xs.dtype)]

        # Expand dimensions for broadcasting
        coords = mols["coords"].unsqueeze(1).repeat(1, len(valid_channels), 1, 1)
        # mask the coords with PADDING_INDEX
        mask = mols["atoms_channel"].unsqueeze(1) == valid_channels.unsqueeze(0).unsqueeze(2)
        coords[~mask] = torch.tensor(PADDING_INDEX, dtype=xs.dtype)

        # Compute distance between x and each atom and the exponent
        # 使用逆二次函数替换高斯函数
        s = 0.825  # 调整参数，使在 0.93r 处的值与原高斯相同
        dist = torch.cdist(xs.unsqueeze(1), coords, p=2)
        exponent = dist / (s * self.radius)

        # Mask to avoid computation when exponent > 10
        valid_mask = exponent <= 10
        exponent = torch.clamp(exponent, max=10)

        # Compute occupancy for the channels
        exp_term = 1 / (1 + exponent ** 2)  # Lorentzian型公式
        log_term = torch.log(1 - exp_term)
        log_term[~valid_mask] = 0  # Set log_term to 0 where exponent > 10 to avoid unnecessary computation
        occs[:, :, valid_channels] = (1 - torch.exp(log_term.sum(3).transpose(1, 2)))

        return occs

    def set_sample_points(self, sample_points: bool):
        self.sample_points = sample_points

# <The following voxel computation code is highly inspired by https://bitbucket.org/grogdrinker/pyuul/src/master/pyuul/VolumeMaker.py and covered by the LGPLv3 license >
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
class Voxels(torch.nn.Module):  # 用于计算 体素（Voxel）网格，使分子可以用 3D 网格表示
    def __init__(self):
        """
        Initializes the Voxels class.
        """
        super(Voxels, self).__init__()
        self.boxsize = None

    def _transform_coordinates(
        self,
        coords,radius=None):
        """
        Transforms the given coordinates by applying dilatation and translation.

        Parameters:
        coords (numpy.ndarray): The coordinates to be transformed.
        radius (float, optional): An optional radius to be transformed. Defaults to None.

        Returns:
        tuple: A tuple containing the transformed coordinates and, if provided, the transformed radius.
        """
        coords = (coords*self.dilatation) - self.translation
        if radius is not None:
            radius = radius*self.dilatation
            return coords,radius
        else:
            return coords

    def _define_spatial_conformation(
        self,
        mincoords: torch.Tensor,
        cubes_around_atoms_dim: int,
        resolution: float
    )-> None:
        self.translation = (mincoords-(cubes_around_atoms_dim)).unsqueeze(1)
        self.dilatation = 1.0 / resolution

    def forward(
        self,
        coords: torch.Tensor,
        radius: torch.Tensor,
        channels: torch.Tensor,
        numberchannels: int = None,
        resolution: float = 1,
        cubes_around_atoms_dim: int = 5
    )-> torch.Tensor:
        padding_mask = ~channels.eq(torch.tensor(PADDING_INDEX, dtype=channels.dtype))
        if numberchannels is None:
            numberchannels = int(channels[padding_mask].max().cpu().data+1)
        self.featureVectorSize = numberchannels

        arange_type = torch.int16

        gx = torch.arange(-cubes_around_atoms_dim, cubes_around_atoms_dim + 1, device=coords.device, dtype=arange_type)
        gy = torch.arange(-cubes_around_atoms_dim, cubes_around_atoms_dim + 1, device=coords.device, dtype=arange_type)
        gz = torch.arange(-cubes_around_atoms_dim, cubes_around_atoms_dim + 1, device=coords.device, dtype=arange_type)
        self.lato = gx.shape[0]

        x1 = gx.unsqueeze(1).expand(self.lato, self.lato).unsqueeze(-1)
        x2 = gy.unsqueeze(0).expand(self.lato, self.lato).unsqueeze(-1)

        xy = torch.cat([x1, x2], dim=-1).unsqueeze(2).expand(self.lato, self.lato, self.lato, 2)
        x3 = gz.unsqueeze(0).unsqueeze(1).expand(self.lato, self.lato, self.lato).unsqueeze(-1)

        del gx, gy, gz, x1, x2

        self.standard_cube = torch.cat([xy, x3], dim=-1).unsqueeze(0).unsqueeze(0)

        mincoords = torch.min(coords[:, :, :], dim=1)[0]
        mincoords = torch.trunc(mincoords / resolution)
        box_size_x = (math.ceil(torch.max(coords[padding_mask][:,0])/resolution)-mincoords[:,0].min())+(2*cubes_around_atoms_dim+1)
        box_size_y = (math.ceil(torch.max(coords[padding_mask][:,1])/resolution)-mincoords[:,1].min())+(2*cubes_around_atoms_dim+1)
        box_size_z = (math.ceil(torch.max(coords[padding_mask][:,2])/resolution)-mincoords[:,2].min())+(2*cubes_around_atoms_dim+1)

        self._define_spatial_conformation(mincoords,cubes_around_atoms_dim,resolution)	#define the spatial transforms to coordinates
        coords,radius = self._transform_coordinates(coords,radius)

        boxsize = (int(box_size_x),int(box_size_y),int(box_size_z))
        self.boxsize=boxsize

        if max(boxsize)<256:
            self.dtype_indices=torch.uint8
        else:
            self.dtype_indices = torch.int16

        batch = coords.shape[0]
        L = coords.shape[1]

        discrete_coordinates = torch.trunc(coords.data).to(self.dtype_indices)

        radius = radius.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        atNameHashing = channels.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        coords = coords.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        discrete_coordinates = discrete_coordinates.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        distmat_standard_cube = torch.norm(
            coords - ((discrete_coordinates + self.standard_cube + 1) + 0.5 * resolution), dim=-1).to(
            coords.dtype)

        atNameHashing = atNameHashing.long()

        # 使用逆二次函数替换高斯函数
        s = 0.825  # 参数与上述一致
        dist = distmat_standard_cube[padding_mask]
        exponent = dist / (s * radius[padding_mask])

        exp_mask = exponent.ge(10)
        exponent = torch.masked_fill(exponent, exp_mask, 10)
        volume_cubes = 1 / (1 + exponent ** 2)  # Lorentzian型公式

        batch_list = torch.arange(batch, device=coords.device).unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(batch,L,self.lato,self.lato,self.lato)

        cubes_coords = (discrete_coordinates[padding_mask] + self.standard_cube.squeeze(0) + 1)[~exp_mask]
        atNameHashing = atNameHashing[padding_mask].expand(-1,self.lato,self.lato,self.lato)

        volume = torch.zeros(batch, boxsize[0]+1, boxsize[1]+1, boxsize[2]+1, self.featureVectorSize, device=coords.device, dtype=coords.dtype)
        index = (batch_list[padding_mask][~exp_mask].view(-1).long(), cubes_coords[:,0].long(), cubes_coords[:,1].long(), cubes_coords[:,2].long(), atNameHashing[~exp_mask])
        volume_cubes=volume_cubes[~exp_mask].view(-1)

        volume_cubes = torch.log(1 - volume_cubes.contiguous())
        volume = 1- torch.exp(volume.index_put(index,volume_cubes,accumulate=True))
        volume=volume.permute(0,4,1,2,3)
        return volume
