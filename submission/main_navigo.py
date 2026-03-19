import argparse
import json
import os
import time

import anndata
import numpy as np
import torch
from tqdm import tqdm

from navigo.data_utils import check_data, get_dataloader_flow
from navigo.model import MLPTimeGRN, Navigo
from navigo.utils import generate_alignment_cell, matching_forward, vis_log


def parse_arguments():
    parser = argparse.ArgumentParser(description="Navigo training script")
    parser.add_argument("--input_data", required=True, help="Input .h5ad path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--hidden_1", type=int, default=5012, help="Hidden layer width 1")
    parser.add_argument("--hidden_2", type=int, default=5012, help="Hidden layer width 2")
    parser.add_argument("--rounds", type=int, default=10, help="Number of outer training rounds")
    parser.add_argument("--train_steps", type=int, default=200, help="Gradient steps per round")
    parser.add_argument("--flow_steps", type=int, default=10, help="ODE solver steps")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--save_every", type=int, default=1, help="Checkpoint frequency in rounds")
    parser.add_argument("--load_from_pretrained", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device_arg


def load_and_preprocess_data(input_data_path):
    adata = anndata.read(input_data_path)

    if "time" not in adata.obs:
        raise KeyError("Input AnnData must contain `obs['time']`.")
    if "Ms" not in adata.layers or "Mu" not in adata.layers:
        raise KeyError("Input AnnData must contain `layers['Ms']` and `layers['Mu']`.")

    adata_ms = adata.layers["Ms"]
    adata_mu = adata.layers["Mu"]

    adata_m = np.concatenate([adata_ms, adata_mu], axis=1)
    data = torch.tensor(adata_m, dtype=torch.float32)
    data_min = data.amin(dim=0)
    data_max = data.amax(dim=0)
    data = (data - data_min) / (data_max - data_min).clamp_min(1e-7)

    time_label = torch.tensor(np.asarray(adata.obs["time"], dtype=np.float32), dtype=torch.float32)
    return data, time_label


def create_model(input_dim, hidden_1, hidden_2, device):
    model = MLPTimeGRN(input_dim=input_dim, hidden_1=hidden_1, hidden_2=hidden_2).to(device)
    return model


def setup_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def compute_forward_loss(navigo, batch, device):
    z = batch[1].to(device)
    time_label = batch[2].to(device)

    forward_index = torch.where(time_label != time_label.max())[0]
    if forward_index.numel() == 0:
        return None

    alignment_data_forward = batch[4].to(device)
    time_forward = batch[5].to(device)

    z_t, sampled_time, target_s, target_u = navigo.get_train_tuple_sample_flow(
        z0=z[forward_index],
        z1=alignment_data_forward[forward_index],
        time=time_label[forward_index],
        next_time=time_forward[forward_index],
    )

    input_z = torch.cat([z_t, sampled_time], dim=1)
    velocity_s, velocity_u, _, _, _ = navigo.model(input_z)

    time_diff = (time_forward[forward_index] - time_label[forward_index]).unsqueeze(1)
    loss_rectify_s = (target_s - time_diff * velocity_s).pow(2).sum()
    loss_rectify_u = (target_u - time_diff * velocity_u).pow(2).sum()
    total_loss = loss_rectify_s + loss_rectify_u

    metrics = {
        "loss_rectify_s": loss_rectify_s.detach().item(),
        "loss_rectify_u": loss_rectify_u.detach().item(),
        "velocity_s_norm": torch.mean((velocity_s ** 2).sum(dim=1)).detach().item(),
        "velocity_u_norm": torch.mean((velocity_u ** 2).sum(dim=1)).detach().item(),
    }
    return total_loss, metrics


def train_navigo(navigo, dataloader, optimizer, train_steps, device):
    for _ in tqdm(range(train_steps), desc="Training", ncols=80):
        loss_sum = 0.0
        loss_rectify_s_sum = 0.0
        loss_rectify_u_sum = 0.0
        velocity_s_norm_sum = 0.0
        velocity_u_norm_sum = 0.0
        update_count = 0

        for batch in dataloader:
            result = compute_forward_loss(navigo, batch, device)
            if result is None:
                continue

            loss, metrics = result
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().item()
            loss_rectify_s_sum += metrics["loss_rectify_s"]
            loss_rectify_u_sum += metrics["loss_rectify_u"]
            velocity_s_norm_sum += metrics["velocity_s_norm"]
            velocity_u_norm_sum += metrics["velocity_u_norm"]
            update_count += 1

        divisor = max(update_count, 1)
        vis_log(
            {
                "all_loss": loss_sum / divisor,
                "all_loss_rectify_s": loss_rectify_s_sum / divisor,
                "all_loss_rectify_u": loss_rectify_u_sum / divisor,
                "all_loss_rectify": (loss_rectify_s_sum + loss_rectify_u_sum) / divisor,
                "velocity_s_norm": velocity_s_norm_sum / divisor,
                "velocity_u_norm": velocity_u_norm_sum / divisor,
            }
        )

    return navigo


def save_results(output_dir, alignment_cell, score, round_idx):
    np.savetxt(os.path.join(output_dir, f"alignment_forward_{round_idx}.txt"), alignment_cell)
    with open(os.path.join(output_dir, f"score_{round_idx}.json"), "w", encoding="utf-8") as fp:
        json.dump(score, fp, indent=2)


def main():
    args = parse_arguments()
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    data, time_label = load_and_preprocess_data(args.input_data)

    input_dim = data.shape[1]
    model = create_model(input_dim=input_dim, hidden_1=args.hidden_1, hidden_2=args.hidden_2, device=device)
    navigo = Navigo(model=model, num_steps=args.flow_steps, device=device)

    if args.load_from_pretrained:
        print(f"Loading checkpoint from {args.load_from_pretrained}")
        model.load_state_dict(torch.load(args.load_from_pretrained, map_location=device))

    optimizer = setup_optimizer(model, args.learning_rate)

    alignment_cell = generate_alignment_cell(data, time_label)
    check_data(alignment_cell, time_label)

    for round_idx in tqdm(range(args.rounds), desc="Training Rounds", ncols=80):
        round_start_time = time.time()

        dataloader = get_dataloader_flow(
            data=data,
            time=time_label,
            alignment_cell=alignment_cell,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        navigo = train_navigo(
            navigo=navigo,
            dataloader=dataloader,
            optimizer=optimizer,
            train_steps=args.train_steps,
            device=device,
        )

        alignment_cell, score = matching_forward(navigo, data, time_label, device=device)
        save_results(args.output_dir, alignment_cell, score, round_idx)

        if (round_idx + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{round_idx + 1}.pth")
            torch.save(navigo.model.state_dict(), checkpoint_path)

        check_data(alignment_cell, time_label)
        round_duration = time.time() - round_start_time
        print(f"Round {round_idx + 1}/{args.rounds} completed in {round_duration:.2f}s")

    print("Training completed successfully")


if __name__ == "__main__":
    main()
