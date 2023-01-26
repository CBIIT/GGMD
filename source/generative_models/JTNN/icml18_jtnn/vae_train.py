import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os, sys, math, random, argparse, pickle, rdkit, time, warnings
import numpy as np
from collections import deque
from atomsci.glo.generative_networks.icml18_jtnn.fast_jtnn import *

import logging

logging.basicConfig(format="%(asctime)-15s %(message)s")
alog = logging.getLogger("ATOM")
alog.setLevel(logging.DEBUG)

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


def save_model(model, optimizer, scheduler, args, train_state, loss, fname):
    train_state["lr"] = [x["lr"] for x in optimizer.param_groups if "lr" in x.keys()][0]

    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "hyperparams": args,
            "vocab": train_state["vocab_list"],
            "step": train_state["Total Iter"],
            "epoch": train_state["Epoch"],
            "exact_epoch": train_state["Exact Epoch"],
            "log_msg": args["log_msg"],
            "model_summary": {
                "loss": loss,
                "lr": train_state["lr"],
                "beta": train_state["beta"],
                "kl": train_state["save_meters"][0],
                "word": train_state["save_meters"][1],
                "topo": train_state["save_meters"][2],
                "assm": train_state["save_meters"][3],
                "pnorm": param_norm(model),
                "gnorm": grad_norm(model),
            },
        },
        args["save_dir"] + "/" + fname,
    )


def batch_to_device(batch, device):
    batch = list(batch)
    batch[3] = list(batch[3])
    batch[3][0] = list(batch[3][0])
    batch[1] = tuple(
        [tsr.to(device) if isinstance(tsr, torch.Tensor) else tsr for tsr in batch[1]]
    )
    batch[2] = tuple(
        [tsr.to(device) if isinstance(tsr, torch.Tensor) else tsr for tsr in batch[2]]
    )
    batch[3][0] = tuple(
        [
            tsr.to(device) if isinstance(tsr, torch.Tensor) else tsr
            for tsr in batch[3][0]
        ]
    )
    batch[3][1] = batch[3][1].to(device)
    batch[3][0] = tuple(batch[3][0])
    batch[3] = tuple(batch[3])
    batch = tuple(batch)

    return batch


# Start of main function

parser = argparse.ArgumentParser()

parser.add_argument(
    "--train", type=str, required=True
)  # Need to change this to allow model loading
parser.add_argument(
    "--vocab", type=str, required=True
)  # Need to change this to allow model loading
parser.add_argument(
    "--save_dir", type=str, required=True
)  # For model loading, this can be ignored, but need to add logic
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--log_msg", type=str, default="")

parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--print_iter", type=int, default=100)

parser.add_argument("--save_by", type=str, default="iter", choices=["iter", "epoch"])
parser.add_argument("--save_interval", type=int, default=5000)
parser.add_argument("--data_parallel", action="store_true")
parser.add_argument("--debug", action="store_true", help="Turn off error catching")

# Network size hyperparameters
parser.add_argument("--hidden_size", type=int, default=450)
parser.add_argument("--latent_size", type=int, default=56)
parser.add_argument("--depthT", type=int, default=20, help="Depth of tree networks")
parser.add_argument("--depthG", type=int, default=3, help="Depth of graph networks")

# Training hyperparameters
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--clip_norm", type=float, default=50.0)
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--step_beta", type=float, default=0.001)
parser.add_argument("--max_beta", type=float, default=1.0)
parser.add_argument("--warmup", type=int, default=40000)
parser.add_argument("--anneal_rate", type=float, default=0.9)
parser.add_argument("--anneal_iter", type=int, default=40000)
parser.add_argument("--kl_anneal_iter", type=int, default=1000)
parser.add_argument("--shuffle", type=bool, default=False)

args = vars(parser.parse_args())
print(args)

# Initate model vocabulary and training parameter dictionary
vocab_list = pd.read_csv(args["vocab"], header=None).values.squeeze()
vocab = Vocab(vocab_list)

train_state = {
    "Epoch": 0,
    "Exact Epoch": 0,
    "Total Iter": 0,
    "beta": args["beta"],
    "n_batches": np.NaN,
    "vocab_list": vocab_list,
    "Iter Start Time": 0,
    "print_meters": np.zeros(4),
    "save_meters": np.zeros(4),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Either initiate or load model
if args["load_path"] is not None:
    warnings.warn("Loading models is not tested")

    checkpoint = torch.load(args["load_path"])

    for key in checkpoint["hyperparams"]:
        args[key] = checkpoint["hyperparams"][key]

    train_state["Epoch"] = checkpoint["epoch"] + 1
    train_state["Exact Epoch"] = checkpoint["exact_epoch"] + 1
    train_state["Total Iter"] = checkpoint["step"]

    model = JTNNVAE(
        vocab,
        args["hidden_size"],
        args["latent_size"],
        args["depthT"],
        args["depthG"],
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = lr_scheduler.ExponentialLR(optimizer, args["anneal_rate"])
    scheduler.step()
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print("Using model training settings, all input ignored")
else:
    model = JTNNVAE(
        vocab,
        args["hidden_size"],
        args["latent_size"],
        args["depthT"],
        args["depthG"],
        device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    scheduler = lr_scheduler.ExponentialLR(optimizer, args["anneal_rate"])
    scheduler.step()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

model.to(device)
model.train()
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
print(model)

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(
    sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None])
)

# Initiate data parallelism
if args["data_parallel"]:
    warnings.warn("Data parallel currently throws a cuda assert error")

    # model.jtnn.GRU.to(device)
    # model.jtnn.to(device)
    # model.decoder.to(device)
    # model.jtmpn.to(device)
    # model.mpn.to(device)

    # wz   = nn.DataParallel(model.jtnn.GRU.W_z)
    # gru   = nn.DataParallel(model.jtnn.GRU)
    # jtnn  = nn.DataParallel(model.jtnn)
    # dec   = nn.DataParallel(model.decoder)
    # jtmpn = nn.DataParallel(model.jtmpn)
    # mpn   = nn.DataParallel(model.mpn)
    model = nn.DataParallel(model)

# Initiate data loader and calculate number of batches for epoch calculation
loader = MolTreeFolder(
    args["train"],
    vocab,
    args["batch_size"],
    num_workers=args["workers"],
    shuffle=args["shuffle"],
)
for batch in loader:
    train_state["n_batches"] += 1

for epoch in range(train_state["Epoch"], args["epoch"] + train_state["Epoch"]):
    print("Epoch: " + str(epoch))
    train_state["Epoch"] = epoch
    train_state["Iter Start Time"] = time.time()
    sys.stdout.flush()
    alog.info(
        "%7s %6s %7s %8s %8s %7s %7s %7s %7s %7s %11s"
        % (
            "Iter#",
            "Loss",
            "LR",
            "Beta",
            "KL",
            "Word",
            "Topo",
            "Assm",
            "PNorm",
            "GNorm",
            "Iter Time",
        )
    )

    if (
        (args["save_by"] == "epoch")
        and (epoch % args["save_interval"] == 0)
        and (epoch > 0)
    ):
        train_state["save_meters"] /= args["save_interval"] * args["batch_size"]
        save_model(
            model,
            optimizer,
            scheduler,
            args,
            train_state,
            loss.cpu().detach().numpy(),
            "model.epoch-" + str(train_state["Epoch"]),
        )
        train_state["save_meters"] *= 0

    for batch in loader:
        batch = batch_to_device(batch, device)

        train_state["Total Iter"] += 1
        train_state["Exact Epoch"] = np.round(
            train_state["Total Iter"] / train_state["n_batches"], 2
        )
        try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc = model(batch, train_state["beta"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args["clip_norm"])
            optimizer.step()
        except Exception as e:
            if args["debug"]:
                raise (e)
            else:
                print("Error: " + str(e))

        train_state["print_meters"] += np.asarray(
            [kl_div, wacc * 100, tacc * 100, sacc * 100], dtype=np.float32
        )
        train_state["save_meters"] += np.asarray(
            [kl_div, wacc * 100, tacc * 100, sacc * 100], dtype=np.float32
        )

        # time to print meters, check within this loop if it also time to checkpoint
        if train_state["Total Iter"] % args["print_iter"] == 0:
            train_state["print_meters"] /= args["print_iter"]
            train_state["lr"] = [
                x["lr"] for x in optimizer.param_groups if "lr" in x.keys()
            ][0]
            sys.stdout.flush()
            alog.info(
                "[%5i] %6.2f %7.1e  %7.1e  %7.2f  %6.2f  %6.2f  %6.2f  %6.2f  %6.2f  %6.1f min "
                % (
                    train_state["Total Iter"],
                    loss.cpu().data.numpy(),
                    train_state["lr"],
                    train_state["beta"],
                    train_state["print_meters"][0],
                    train_state["print_meters"][1],
                    train_state["print_meters"][2],
                    train_state["print_meters"][3],
                    param_norm(model),
                    grad_norm(model),
                    (time.time() - train_state["Iter Start Time"]) / 60,
                )
            )

            # Reset iter start time, zero meters, force stdout
            train_state["Iter Start Time"] = time.time()
            train_state["print_meters"] *= 0
            sys.stdout.flush()

        # Save model if the iteration is correct
        if (args["save_by"] == "iter") and (
            train_state["Total Iter"] % args["save_interval"] == 0
        ):
            train_state["save_meters"] /= args["save_interval"]
            save_model(
                model,
                optimizer,
                scheduler,
                args,
                train_state,
                loss.cpu().detach().numpy(),
                "model.iter-" + str(train_state["Total Iter"]),
            )
            train_state["save_meters"] *= 0

        if train_state["Total Iter"] % args["anneal_iter"] == 0:
            scheduler.step()
            alog.info("learning rate: %.6f" % scheduler.get_lr()[0])

        if (train_state["Total Iter"] % args["kl_anneal_iter"] == 0) and (
            train_state["Total Iter"] >= args["warmup"]
        ):
            train_state["beta"] = min(
                args["max_beta"], train_state["beta"] + args["step_beta"]
            )
            print("KL Annealing beta increased to " + str(train_state["beta"]))
