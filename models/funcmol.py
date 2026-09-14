import math
import os

import torch
from torch import nn
from tqdm import tqdm
from funcmol.utils.utils_fm import add_noise_to_code
from funcmol.utils.utils_base import save_xyz, convert_xyzs_to_sdf
from funcmol.models.unet1d import MLPResCode


########################################################################################
# create funcmol
def create_funcmol(config: dict, fabric: object):
    """
    Create and compile a FuncMol model.

    Args:
        config (dict): Configuration dictionary for the FuncMol model.
        fabric (object): An object providing necessary methods and attributes for model creation.

    Returns:
        torch.nn.Module: The compiled FuncMol model.
    """
    model = FuncMol(config, fabric=fabric)

    # n params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print(f">> FuncMol has {(n_params/1e6):.02f}M parameters")

    #model = torch.compile(model)

    return model


########################################################################################
# FuncMol class
class FuncMol(nn.Module):
    def __init__(self, config: dict, fabric):
        super().__init__()
        self.device = fabric.device
        self.smooth_sigma = config["smooth_sigma"]

        # denoiser model
        self.net = MLPResCode(
            code_dim=config["decoder"]["code_dim"],
            n_hidden_units=config["denoiser"]["n_hidden_units"],
            num_blocks=config["denoiser"]["num_blocks"],
            n_groups=config["denoiser"]["n_groups"],
            dropout=config["denoiser"]["dropout"],
            bias_free=config["denoiser"]["bias_free"],
        )

    def forward(self, y: torch.Tensor):
        """
        Forward pass of the denoiser model.

        Args:
            y (torch.Tensor): Input tensor of shape (batch_size, channel_size, c_size).

        Returns:
            torch.Tensor: Output tensor after passing through the denoiser model.
        """
        return self.net(y)

    def score(self, y: torch.Tensor):
        """
        Calculates the score of the denoiser model.

        Args:
        - y: Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - score: The score tensor of shape (batch_size, channels, height, width).
        """
        xhat = self.forward(y)
        return (xhat - y) / (self.smooth_sigma**2)

    @torch.no_grad()
    def wjs_walk_steps(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_steps: int,
        delta: float = 0.5,
        friction: float = 1.0,
        lipschitz: float = 1.0,
        scheme: str = "aboba",
        temperature: float = 1.0
    ):
        """
        Perform a series of steps using the Weighted Jump Stochastic (WJS) method.

        Args:
            q (torch.Tensor): The initial position tensor.
            p (torch.Tensor): The initial momentum tensor.
            n_steps (int): The number of steps to perform.
            delta (float, optional): The time step size. Default is 0.5.
            friction (float, optional): The friction coefficient. Default is 1.0.
            lipschitz (float, optional): The Lipschitz constant. Default is 1.0.
            scheme (str, optional): The integration scheme to use, either "aboba" or "baoab". Default is "aboba".
            temperature (float, optional): The temperature parameter. Default is 1.0.

        Returns:
            tuple: A tuple containing the updated position tensor `q` and the updated momentum tensor `p`.
        """
        u = pow(lipschitz, -1)  # inverse mass
        delta *= self.smooth_sigma
        zeta1 = math.exp(-friction * delta)
        zeta2 = math.exp(-2 * friction * delta)
        if scheme == "aboba":
            for _ in range(n_steps):
                q += delta * p / 2  # q_{t+1/2}
                psi = self.score(q)
                p += u * delta * psi / 2  # p_{t+1}
                p = (
                    zeta1 * p + u * delta * psi / 2 + math.sqrt(temperature) * math.sqrt(u * (1 - zeta2)) * torch.randn_like(q)
                )  # p_{t+1}
                q += delta * p / 2  # q_{t+1}
        elif scheme == "baoab":
            for _ in range(n_steps):
                p += u * delta * self.score(q) / 2  # p_{t+1/2}
                q += delta * p / 2  # q_{t+1/2}
                phat = zeta1 * p + math.sqrt(temperature) * math.sqrt(u * (1 - zeta2)) * torch.randn_like(q)  # phat_{t+1/2}
                q += delta * phat / 2  # q_{t+1}
                psi = self.score(q)
                p = phat + u * delta * psi / 2  # p_{t+1}
        return q, p

    @torch.no_grad()
    def wjs_jump_step(self, y: torch.Tensor):
        """Jump step of walk-jump sampling.
        Recover clean sample x from noisy sample y.
        It is a simple forward of the network.

        Args:
            y (torch.Tensor): samples y from mcmc chain


        Returns:
            torch.Tensor: estimated ``clean'' samples xhats
        """
        return self.forward(y)

    def initialize_y_v(
        self,
        n_chains: int = 25,
        code_dim: int = 1024,
        code_stats: dict = None,
    ):
        """
        Initializes the latent variable `y` with uniform noise and adds Gaussian noise.

        Args:
            n_chains (int, optional): Number of chains to initialize. Defaults to 25.
            code_dim (int, optional): Dimensionality of the code. Defaults to 1024.
            code_stats (dict, optional): Dictionary containing the minimum and maximum
                                         normalized values for the uniform noise.
                                         Defaults to None.

        Returns:
            tuple: A tuple containing:
                - y (torch.Tensor): Tensor of shape (n_chains, code_dim) with added Gaussian noise.
                - torch.Tensor: Tensor of zeros with the same shape as `y`.
        """
        # uniform noise
        y = torch.empty((n_chains, code_dim), device=self.device, dtype=torch.float32).uniform_(code_stats["min_normalized"], code_stats["max_normalized"])

        # gaussian noise
        y = add_noise_to_code(y, self.smooth_sigma)

        return y, torch.zeros_like(y)

    def sample(
            self,
            dec: object,
            save_dir: str,
            config: dict,
            fabric=None,
            delete_net: bool = False,
    ):
        """
        Sample molecular codes using WJS, convert to molecules, save in XYZ/SDF.
        """
        self.eval()
        os.makedirs(f"{save_dir}/figures", exist_ok=True)

        n_atoms_per_chain = config["wjs"].get("n_atoms", 20)

        # ----------------------------
        # Step 1: Sample codes with WJS
        # ----------------------------
        codes_all = []
        fabric.print(f">> Sample codes with WJS (n_chains: {config['wjs']['n_chains']})")
        for rep in range(config["wjs"]["repeats_wjs"]):
            y, v = self.initialize_y_v(
                n_chains=config["wjs"]["n_chains"],
                code_dim=config["decoder"]["code_dim"],
                code_stats=dec.code_stats,
            )
            for _ in range(0, config["wjs"]["max_steps_wjs"], config["wjs"]["steps_wjs"]):
                y, v = self.wjs_walk_steps(
                    y, v,
                    config["wjs"]["steps_wjs"],
                    delta=config["wjs"]["delta_wjs"],
                    friction=config["wjs"]["friction_wjs"]
                )
                code_hats = self.wjs_jump_step(y)
                codes_all.append(code_hats.cpu())

        if delete_net:
            del self.net
            torch.cuda.empty_cache()

        codes = torch.cat(codes_all, dim=0)

        # ----------------------------
        # Step 2: Batch render molecules
        # ----------------------------
        batch_size_render_codes = min(codes.size(0), config["wjs"]["batch_size_render"])
        batched_codes = torch.split(codes, batch_size_render_codes, dim=0)
        mols = []
        fabric.print(f">> Rendering molecules - batches of {batch_size_render_codes}")
        for batched_code in batched_codes:
            mols += dec.codes_to_molecules(
                batched_code,
                unnormalize=config["normalize_codes"],
                fabric=fabric,
                config=config,
                n_atoms=n_atoms_per_chain
            )

        # ----------------------------
        # Step 3: Save molecules
        # ----------------------------
        save_dir = os.path.join(os.getcwd(), save_dir)
        molecules_xyz = save_xyz(mols, save_dir, fabric, atom_elements=config["dset"]["elements"])
        convert_xyzs_to_sdf(save_dir, fabric=fabric)

        del mols, codes, batched_codes, codes_all
        return molecules_xyz

