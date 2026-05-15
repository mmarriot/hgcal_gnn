import os
import os.path as osp
import glob

import numpy as np
import uproot
import awkward as ak

import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

### to be honest, i think this is not needed, think it works without
def find_highest_branch(path, base_name):
    with uproot.open(path) as f:
        branches = [k for k in f.keys() if k.startswith(base_name + ';')]
        sorted_branches = sorted(branches, key=lambda x: int(x.split(';')[-1]))
        return sorted_branches[-1] if sorted_branches else None


class CCV1(Dataset):
    """
    Loads trackster-level features and CaloParticle associations from ROOT files.

    Each event becomes one Data object with:
        x       : (N, 16) trackster features
        assoc   : (N,)    group ID per trackster (index of best-matched CaloParticle)
        scores  : (N, 4)  reco-to-sim scores for up to 4 associations
        links   : (N, 4)  CaloParticle indices for up to 4 associations

    Events with fewer than 2 CaloParticles or fewer than 2 trackster associations
    are skipped during loading.
    """

    url = '/dummy/'

    def __init__(self, root, transform=None, max_events=1e8, inp='train'):
        super(CCV1, self).__init__(root, transform)
        self.inp = inp
        self.max_events = max_events
        self.fill_data(max_events)

    def fill_data(self, max_events):
        counter = 0
        print("### Loading tracksters data")

        for path in tqdm(self.raw_paths):
            print(path)

            tracksters_path  = find_highest_branch(path, 'tracksters')
            associations_path = find_highest_branch(path, 'associations')
            simtrack          = find_highest_branch(path, 'simtrackstersCP')

            for array in uproot.iterate(
                f"{path}:{tracksters_path}",
                [
                    "time", "raw_energy",
                    "barycenter_x", "barycenter_y", "barycenter_z",
                    "barycenter_eta", "barycenter_phi",
                    "EV1", "EV2", "EV3",
                    "eVector0_x", "eVector0_y", "eVector0_z",
                    "sigmaPCA1", "sigmaPCA2", "sigmaPCA3",
                    "raw_pt", "vertices_time",
                ],
            ):
                tmp_time         = array["time"]
                tmp_raw_energy   = array["raw_energy"]
                tmp_bx           = array["barycenter_x"]
                tmp_by           = array["barycenter_y"]
                tmp_bz           = array["barycenter_z"]
                tmp_beta         = array["barycenter_eta"]
                tmp_bphi         = array["barycenter_phi"]
                tmp_EV1          = array["EV1"]
                tmp_EV2          = array["EV2"]
                tmp_EV3          = array["EV3"]
                tmp_eV0x         = array["eVector0_x"]
                tmp_eV0y         = array["eVector0_y"]
                tmp_eV0z         = array["eVector0_z"]
                tmp_sigma1       = array["sigmaPCA1"]
                tmp_sigma2       = array["sigmaPCA2"]
                tmp_sigma3       = array["sigmaPCA3"]
                tmp_pt           = array["raw_pt"]
                tmp_vt           = array["vertices_time"]

                vert_array = []
                for vert_chunk in uproot.iterate(f"{path}:{simtrack}", ["barycenter_x"]):
                    vert_array = vert_chunk["barycenter_x"]
                    break

                tmp_assoc = []
                score_array = []
                for assoc_chunk in uproot.iterate(
                    f"{path}:{associations_path}",
                    ["tsCLUE3D_recoToSim_CP", "tsCLUE3D_recoToSim_CP_score"],
                ):
                    tmp_assoc   = assoc_chunk["tsCLUE3D_recoToSim_CP"]
                    score_array = assoc_chunk["tsCLUE3D_recoToSim_CP_score"]
                    break

                # Require >= 2 CaloParticles per event.
                skim_mask = [len(e) >= 2 for e in vert_array]

                def apply_mask(arrays, mask):
                    return [a[mask] for a in arrays]

                all_arrays = [
                    tmp_time, tmp_raw_energy, tmp_bx, tmp_by, tmp_bz,
                    tmp_beta, tmp_bphi, tmp_EV1, tmp_EV2, tmp_EV3,
                    tmp_eV0x, tmp_eV0y, tmp_eV0z,
                    tmp_sigma1, tmp_sigma2, tmp_sigma3,
                    tmp_assoc, tmp_pt, tmp_vt, score_array,
                ]
                (
                    tmp_time, tmp_raw_energy, tmp_bx, tmp_by, tmp_bz,
                    tmp_beta, tmp_bphi, tmp_EV1, tmp_EV2, tmp_EV3,
                    tmp_eV0x, tmp_eV0y, tmp_eV0z,
                    tmp_sigma1, tmp_sigma2, tmp_sigma3,
                    tmp_assoc, tmp_pt, tmp_vt, score_array,
                ) = [a[skim_mask] for a in all_arrays]

                # Require >= 2 particles  associations per event.
                skim_mask2 = [len(e) >= 2 for e in tmp_assoc]
                (
                    tmp_time, tmp_raw_energy, tmp_bx, tmp_by, tmp_bz,
                    tmp_beta, tmp_bphi, tmp_EV1, tmp_EV2, tmp_EV3,
                    tmp_eV0x, tmp_eV0y, tmp_eV0z,
                    tmp_sigma1, tmp_sigma2, tmp_sigma3,
                    tmp_assoc, tmp_pt, tmp_vt, score_array,
                ) = [a[skim_mask2] for a in [
                    tmp_time, tmp_raw_energy, tmp_bx, tmp_by, tmp_bz,
                    tmp_beta, tmp_bphi, tmp_EV1, tmp_EV2, tmp_EV3,
                    tmp_eV0x, tmp_eV0y, tmp_eV0z,
                    tmp_sigma1, tmp_sigma2, tmp_sigma3,
                    tmp_assoc, tmp_pt, tmp_vt, score_array,
                ]]

                if counter == 0:
                    self.time        = tmp_time
                    self.raw_energy  = tmp_raw_energy
                    self.bx          = tmp_bx
                    self.by          = tmp_by
                    self.bz          = tmp_bz
                    self.beta        = tmp_beta
                    self.bphi        = tmp_bphi
                    self.EV1         = tmp_EV1
                    self.EV2         = tmp_EV2
                    self.EV3         = tmp_EV3
                    self.eV0x        = tmp_eV0x
                    self.eV0y        = tmp_eV0y
                    self.eV0z        = tmp_eV0z
                    self.sigma1      = tmp_sigma1
                    self.sigma2      = tmp_sigma2
                    self.sigma3      = tmp_sigma3
                    self.assoc       = tmp_assoc
                    self.pt          = tmp_pt
                    self.vt          = tmp_vt
                    self.score       = score_array
                else:
                    self.time        = ak.concatenate((self.time,       tmp_time))
                    self.raw_energy  = ak.concatenate((self.raw_energy, tmp_raw_energy))
                    self.bx          = ak.concatenate((self.bx,         tmp_bx))
                    self.by          = ak.concatenate((self.by,         tmp_by))
                    self.bz          = ak.concatenate((self.bz,         tmp_bz))
                    self.beta        = ak.concatenate((self.beta,        tmp_beta))
                    self.bphi        = ak.concatenate((self.bphi,        tmp_bphi))
                    self.EV1         = ak.concatenate((self.EV1,         tmp_EV1))
                    self.EV2         = ak.concatenate((self.EV2,         tmp_EV2))
                    self.EV3         = ak.concatenate((self.EV3,         tmp_EV3))
                    self.eV0x        = ak.concatenate((self.eV0x,        tmp_eV0x))
                    self.eV0y        = ak.concatenate((self.eV0y,        tmp_eV0y))
                    self.eV0z        = ak.concatenate((self.eV0z,        tmp_eV0z))
                    self.sigma1      = ak.concatenate((self.sigma1,      tmp_sigma1))
                    self.sigma2      = ak.concatenate((self.sigma2,      tmp_sigma2))
                    self.sigma3      = ak.concatenate((self.sigma3,      tmp_sigma3))
                    self.assoc       = ak.concatenate((self.assoc,       tmp_assoc))
                    self.pt          = ak.concatenate((self.pt,          tmp_pt))
                    self.vt          = ak.concatenate((self.vt,          tmp_vt))
                    self.score       = ak.concatenate((self.score,       score_array))

                counter += len(tmp_bx)
                if counter >= max_events:
                    print(f"Reached {max_events} events!")
                    break
            if counter >= max_events:
                break

    def download(self):
        raise RuntimeError(
            f'Dataset not found. Please download it from {self.url} and move all '
            f'*.root files to {self.raw_dir}')

    def len(self):
        return len(self.time)

    @property
    def raw_file_names(self):
        return sorted(glob.glob(osp.join(self.raw_dir, '*.root')))

    @property
    def processed_file_names(self):
        return []

    def get(self, idx):

        def ensure_four_columns(tensor):
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
            nrow, ncol = tensor.shape
            if ncol > 4:
                return tensor[:, :4]
            elif ncol < 4:
                last_col = tensor[:, -1].unsqueeze(1)
                return torch.cat([tensor, last_col.repeat(1, 4 - ncol)], dim=1)
            return tensor

        def reconstruct_array(grouped_indices):
            max_index = max(max(indices) for indices in grouped_indices.values())
            reconstructed = [-1] * (max_index + 1)
            for value, indices in grouped_indices.items():
                for idx2 in indices:
                    reconstructed[idx2] = value
            return reconstructed

        flat_feats = np.column_stack((
            np.array(self.bx[idx]),    np.array(self.by[idx]),    np.array(self.bz[idx]),
            np.array(self.raw_energy[idx]),
            np.array(self.beta[idx]),  np.array(self.bphi[idx]),
            np.array(self.EV1[idx]),   np.array(self.EV2[idx]),   np.array(self.EV3[idx]),
            np.array(self.eV0x[idx]),  np.array(self.eV0y[idx]),  np.array(self.eV0z[idx]),
            np.array(self.sigma1[idx]), np.array(self.sigma2[idx]), np.array(self.sigma3[idx]),
            np.array(self.pt[idx]),
        ))
        x = torch.from_numpy(flat_feats).float()

        links_tensor  = torch.from_numpy(np.array(self.assoc[idx]).astype(np.int64))
        scores_tensor = torch.from_numpy(np.array(self.score[idx])).float()

        scores_tensor = ensure_four_columns(scores_tensor)
        links_tensor  = ensure_four_columns(links_tensor)

        # Assign group ID = CaloParticle index with the smallest (best) score per trackster.
        total_tracksters = x.size(0)
        new_assoc = []
        for i in range(total_tracksters):
            min_index = torch.argmin(scores_tensor[i]).item()
            new_assoc.append(int(links_tensor[i, min_index].item()))

        groups = {}
        for i, g in enumerate(new_assoc):
            groups.setdefault(g, []).append(i)
        assoc_array = reconstruct_array(groups)

        return Data(x=x, assoc=assoc_array, scores=scores_tensor, links=links_tensor)
