import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import matplotlib.pyplot as plt
from IPython.display import clear_output

from .utils import mIoU


def ddp_setup(rank: int, world_size: int):

    """
    Parameters
    ----------
    rank: int 
        Unique identifier of each process

    world_size: int 
        Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:

    """
    Trainer class for DDP model

    Parameters
    ----------

    model: nn.Module

    train_data: Dataloader

    val_data: DataLoader

    test_data: DataLoader

    optimizer: torch.optim.Optimizer

    scheduler: torch Scheduler
        training scheduler
    gpu_id: int
        Unique identifier of each gpu
    title: str
        title of plot if plot = True
    snapshot_path: str
        path to save best model
    plot: bool
        whether to plot loss and mIoU during training or not
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        test_data: DataLoader,
        optimizer: Optimizer,
        scheduler,
        gpu_id: int,
        title: str,
        snapshot_path: str,
        plot: bool
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.title = title
        self.loss_fn = nn.CrossEntropyLoss()

        torch.cuda.set_device(gpu_id)

        self.model = DDP(model, device_ids=[gpu_id])

        self.train_miou = []
        self.val_miou = []
        self.val_loss = []
        self.train_loss = []
        self.test_miou = 0
        self.best_val_miou = 0.0
        self.best_state = model.state_dict()
        self.snapshot_path = snapshot_path
        self.plot = plot
        # In order to make plots readable I manually clip the loss if it exceeds MAX_LOSS
        self.MAX_LOSS = 5

    def _plot_metrics(self, epoch):

        """Plots loss and mIoU every epoch"""

        clear_output(wait=True)

        plt.rcParams["figure.figsize"] = (15, 8)

        plt.subplot(1, 2, 1)
        plt.plot(range(epoch), self.train_miou, label='train miou')
        plt.plot(range(epoch), self.val_miou, label='validation miou')
        plt.title(label=f'{self.title} miou')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        self.val_loss[-1] = self.MAX_LOSS if self.val_loss[-1] > self.MAX_LOSS else self.val_loss[-1]
        self.train_loss[-1] = self.MAX_LOSS if self.train_loss[-1] > self.MAX_LOSS else self.train_loss[-1]
        plt.plot(range(epoch), self.train_loss, label='train loss')
        plt.plot(range(epoch), self.val_loss, label='validation loss')
        plt.title(label=f'{self.title} loss')
        plt.grid()
        plt.legend()

        plt.show()

    def _train_epoch(self, epoch):

        """
        Trains model for one epoch

        Parameters
        ----------
        epoch: int
            epoch number

        Returns
        -------
            mean_epoch_loss: np.array
            mean_epoch_miou: np.array
        """

        self.model.train()
        self.train_data.sampler.set_epoch(epoch)
        epoch_loss = []
        epoch_miou = []
        with tqdm(
            enumerate(self.train_data),
            total=len(self.train_data),
            desc='Training',
            disable=self.gpu_id != 0
        ) as tqdm_train:
            for _, (images, labels) in tqdm_train:
                self.optimizer.zero_grad()
                images = images.to(self.gpu_id)
                labels = labels.to(self.gpu_id)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels.squeeze().long())

                epoch_loss.append(loss.item())
                epoch_miou.append(mIoU(outputs, labels))

                loss.backward()
                self.optimizer.step()

                tqdm_train.set_postfix(
                    loss=np.mean(epoch_loss), miou=np.mean(epoch_miou)
                )
        return np.mean(epoch_loss), np.mean(epoch_miou)

    def _val_epoch(self):

        """Validates model after _train_epoch"""

        self.model.module.eval()
        val_loss = []
        val_miou = []

        with tqdm(
            enumerate(self.val_data), total=len(self.val_data), desc='Validating'
        ) as tqdm_val:
            with torch.no_grad():
                for _, (images, labels) in tqdm_val:
                    images = images.to(self.gpu_id)
                    labels = labels.to(self.gpu_id)

                    outputs = self.model.module(images)
                    loss = self.loss_fn(outputs, labels.squeeze().long())

                    val_loss.append(loss.item())
                    val_miou.append(mIoU(outputs, labels))

                    tqdm_val.set_postfix(
                        val_loss=np.mean(val_loss),
                        val_miou=np.mean(val_miou)
                    )
            return np.mean(val_loss), np.mean(val_miou)

    def _evaluate(self) -> int:

        """
        Evaluates model after whole training

        Returns
        -------
        mean_miou: int
            mean miou across all batches in test dataloader
        """

        self.model.module.load_state_dict(self.best_state)
        self.model.module.eval()

        eval_miou = []

        with tqdm(
            enumerate(self.test_data), total=len(self.test_data), desc='Testing'
        ) as tqdm_test:
            with torch.no_grad():
                for _, (images, labels) in tqdm_test:
                    images = images.to(self.gpu_id)
                    labels = labels.to(self.gpu_id)

                    outputs = self.model.module(images)

                    miou = mIoU(outputs, labels)
                    eval_miou.append(miou)
                    tqdm_test.set_postfix(val_miou=np.mean(miou))

        return np.mean(eval_miou)


    def train(self, max_epochs: int):

        """
        Main training function that runs _train_epoch, _val_epoch and plots if needed

        Parameters
        ----------

        max_epochs: int
            number of training epochs
        """

        for epoch in range(max_epochs):

            if self.gpu_id == 0:
                print(f'Epoch: [{epoch+1}/{max_epochs}]')
            epoch_train_loss, epoch_train_miou = self._train_epoch(epoch)

            if self.gpu_id == 0:
                epoch_val_loss, epoch_val_miou = self._val_epoch()

                self.train_miou.append(epoch_train_miou)
                self.val_miou.append(epoch_val_miou)
                self.val_loss.append(epoch_val_loss)
                self.train_loss.append(epoch_train_loss)

                if self.plot:
                    self._plot_metrics(epoch+1)

                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(epoch_val_loss)
                    else:
                        self.scheduler.step()

                if epoch_val_miou > self.best_val_miou:
                    self.best_val_miou = epoch_val_miou
                    self.best_state = self.model.module.state_dict()

            if self.gpu_id == 0:
                print(f'Validation mIoU: {epoch_val_miou:.3f}, \
                      best_val_mIoU: {self.best_val_miou:.3f}')

        if self.gpu_id == 0:
            self.test_miou = self._evaluate()
            torch.save(self.best_state, self.snapshot_path)

    def get_results(self):

        """Returns all results after training not including model state_dict"""

        return {
            'best_val_miou': self.best_val_miou,
            'train_miou': self.train_miou,
            'val_miou': self.val_miou,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'test_miou': self.test_miou
        }


def prepare_dataloader(dataset: Dataset, batch_size: int, ddp_sampler=True):
    """Casts Dataloader to Distributed Dataloader if needed"""

    if ddp_sampler:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=16,
        sampler=sampler
    )


def main(
    rank: int,
    world_size: int,
    model: nn.Module,
    total_epochs: int,
    batch_size: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    optimizer,
    scheduler,
    sender,
    title: str='title',
    snapshot_path='snapshot',
    plot=False
):
    """
    Main function that launches Trainer


    Parameters
    ----------

    rank: int
        Unique identifier of each process

    world_size: int 
        Total number of processes

    model: nn.Module
        model to train

    total_epochs: int

    train_dataset: Dataset

    val_dataset: Dataset

    test_dataset: Dataset

    optimizer: torch.optim.Optimizer

    scheduler: torch Scheduler

    sender: mp.Pipe sender
        mp.Pipe sender object to send the training results after training is over

    title: str
        title of the plot if plot = True

    snapshot_path: str
        path where model state dict will be stored

    plot: bool
        whether to plot loss and mIoU during training


    Returns(via sender)
    -------
        dict of 
            best_val_miou: float
            train_miou: List[float]
            val_miou: List[float]
            train_loss: List[float]
            val_loss: List[float]
            test_miou: float
    """
    ddp_setup(rank, world_size)

    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size, ddp_sampler=False)
    test_data = prepare_dataloader(test_dataset, batch_size, ddp_sampler=False)

    trainer = Trainer(
        model = model,
        train_data = train_data,
        val_data = val_data,
        test_data = test_data,
        optimizer = optimizer,
        scheduler = scheduler,
        gpu_id = rank,
        title = title,
        snapshot_path = snapshot_path,
        plot = plot
    )
    trainer.train(total_epochs)
    results = trainer.get_results()
    if rank == 0:
        sender.send(results)
    destroy_process_group()
