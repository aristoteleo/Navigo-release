from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch


class NavigoDataset(Dataset):
    def __init__(self, data, time, alignment_cell):
        self.data = data
        self.time = time
        self.alignment_cell = alignment_cell

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        forward_idx = self.alignment_cell[idx]
        return (
            idx,
            self.data[idx].clone().detach(),
            self.time[idx],
            forward_idx,
            self.data[forward_idx],
            self.time[forward_idx],
        )


def get_dataloader_flow(
    data,
    time,
    alignment_cell,
    batch_size,
    shuffle=True,
    num_workers=0,
    device="cuda",
    use_ddp=False,
):
    del device
    dataset = NavigoDataset(data, time, alignment_cell)
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def check_data(alignment_cell, time_label):
    if isinstance(time_label, torch.Tensor):
        time_np = time_label.detach().cpu().numpy()
    else:
        time_np = np.asarray(time_label)

    time_steps = np.sort(np.unique(time_np))
    for i in range(len(time_steps) - 1):
        index_i = np.where(time_np == time_steps[i])[0]
        if len(index_i) == 0:
            continue
        aligned_times = np.unique(time_np[alignment_cell[index_i]])
        assert len(aligned_times) == 1
        assert aligned_times[0] > time_steps[i]
