import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import numpy as np
import math
from atomsci.glo.generative_networks.icml18_jtnn.fast_jtnn import *
from ray.tune import Trainable
from ray.tune import track


class JTNNTrainable(Trainable):
    # def __init__(self):
    #    super(JTNNTrainable, self).__init__()

    def _setup(self, config):
        # cache the config dict inside of JTNNTrainable for later access

        self.config = config

        # self.num_steps_per_train_iter = 100  # this corresponds to number of mini-batches to iterate over in training loop before returning a validation metric

        self.total_step = 0  # this is a counter that measure how many steps have been taken for this particular Trainable instance, used for checkpointing

        # construct the Vocab object
        self.vocab = [x.strip("\r\n ") for x in open(self.config.get("vocab"))]
        self.vocab = Vocab(self.vocab)

        # instantiate the JTNNVAE model
        self.model = JTNNVAE(
            self.vocab,
            self.config.get("hidden_size"),
            self.config.get("latent_size"),
            self.config.get("depthT"),
            self.config.get("depthG"),
        ).cuda()

        # initialize the model parameters
        for param in self.model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        # instantiate the optimizer and scheduler objects
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.get("lr"))
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer, self.config.get("anneal_rate")
        )

        # take an initial step in the scheduler (need to clarify the need for doing this)
        self.scheduler.step()

        self.train_loader = MolTreeFolder(
            self.config.get("train"),
            self.vocab,
            self.config.get("batch_size"),
            num_workers=1,
        )

        self.val_loader = MolTreeFolder(
            self.config.get("val"),
            self.vocab,
            self.config.get("batch_size"),
            num_workers=1,
        )

        self.beta = self.config.get("beta")

        self.total_step = 0

    def _train(self):
        self.model.train()

        param_norm = lambda m: math.sqrt(
            sum([p.norm().item() ** 2 for p in m.parameters()])
        )
        grad_norm = lambda m: math.sqrt(
            sum(
                [
                    p.grad.norm().item() ** 2
                    for p in m.parameters()
                    if p.grad is not None
                ]
            )
        )

        # for now just keeping track of loss, will look at reconstruction instead for validation set
        loss_list = []
        kl_div_list = []
        wacc_list = []
        tacc_list = []
        sacc_list = []

        for idx, batch in enumerate(self.train_loader):
            # if idx > self.num_steps_per_train_iter:
            # if the number of training steps has been exceeded then return the proper values and exit the loop
            # TODO: if saving model_summary info, make sure those metrics are being cached before exiting, though may only need to be concerned with validation metrics?
            #   break

            try:
                self.model.zero_grad()
                loss, kl_div, wacc, tacc, sacc = self.model(batch, self.beta)

                # collect training metric values
                loss_list.append(loss.cpu().data.numpy())
                kl_div_list.append(kl_div)
                wacc_list.append(wacc)
                tacc_list.append(tacc)
                sacc_list.append(sacc)

                # backpropagation and optimizer step
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.get("clip_norm")
                )
                self.optimizer.step()
                self.total_step += 1

            except RuntimeError as e:
                # print(e)  #don't want to clutter the screen, route to a log file or something?
                continue

            if self.total_step % self.config.get("anneal_iter") == 0:
                self.scheduler.step()

            if self.total_step % self.config.get(
                "kl_anneal_iter"
            ) == 0 and self.total_step >= self.config.get("warmup"):
                self.beta = min(
                    self.config.get("max_beta"),
                    self.beta + self.config.get("step_beta"),
                )

        # insert validation loop? YES and then log the metric
        # track.log(loss=np.mean(loss_list))
        return {
            "loss": np.mean(loss_list),
            "kl_div": np.mean(kl_div_list),
            "wacc": np.mean(wacc_list),
            "tacc": np.mean(tacc_list),
            "sacc": np.mean(sacc_list),
        }

    def _save(self, tmp_checkpoint_dir):
        # TODO: what is the path I'm supposed to use here?
        param_norm = lambda m: math.sqrt(
            sum([p.norm().item() ** 2 for p in m.parameters()])
        )
        grad_norm = lambda m: math.sqrt(
            sum(
                [
                    p.grad.norm().item() ** 2
                    for p in m.parameters()
                    if p.grad is not None
                ]
            )
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "hyperparams": vars(self.config),
                "step": self.total_step,
                "epoch": epoch,
                "log_msg": "hyperparameter optimization",
                "model_summary": None,
            },  # not entirely clear on how I want to store the model_summary fields, either with a None to make things easy but not compataible with other code, or cache the most recent results and use those...need to make sure those correspond to the weights being saved
            "{}/model.iter-{}.pth".format(tmp_checkpoint_dir, str(self.total_step)),
        )

        return "{}/model.iter-{}.pth".format(tmp_checkpoint_dir, str(self.total_step))

    def _restore(self, restore_path):
        # load the torch checkpoint (dictionary)
        checkpoint = torch.load(restore_path)
        # overwrite the state of the optimizer, scheduler, and model
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # TODO: may need to update other metadata
