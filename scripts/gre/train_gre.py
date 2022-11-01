import argparse
from pathlib import Path
from datetime import datetime

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, Accuracy
import torch
from torch.utils import tensorboard
import torch.nn.functional as F

from gpn.RP_dataset import Dataset
from gpn.RP_networks import get_network


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    # create log directory
    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = "net={},batch_size={},lr={:.0e},{}".format(
        time_stamp,
        args.net,
        args.batch_size,
        args.lr,
        args.description,
    ).strip(",")
    logdir = args.logdir / description

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset, args.batch_size, args.val_split, kwargs
    )

    # build the network
    net = get_network(args.net).to(device)

    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    metrics = {
        "loss": Average(lambda out: out[3]),
        "accuracy": Accuracy(lambda out: (torch.round(out[1][0]), out[2][0])),
    }

    # create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(net, loss_fn, metrics, device)

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True).attach(trainer)

    train_writer, val_writer = create_summary_writers(net, device, logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        train_writer.add_scalar("loss", metrics["loss"], epoch)
        train_writer.add_scalar("accuracy", metrics["accuracy"], epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        val_writer.add_scalar("loss", metrics["loss"], epoch)
        val_writer.add_scalar("accuracy", metrics["accuracy"], epoch)

    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        logdir,
        "RP",
        n_saved=100,
        require_empty=True,
        save_as_state_dict=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {args.net: net}
    )

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def create_train_val_loaders(root, batch_size, val_split, kwargs):
    # load the dataset
    dataset = Dataset(root)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs
    )
    return train_loader, val_loader


def prepare_batch(batch, device):
    grasp_pose, label = batch
    grasp_pose = grasp_pose.to(device)
    label = label.float().to(device)
    return grasp_pose, label


def loss_fn(pred, target):
    loss = F.binary_cross_entropy(pred, target)
    return loss


def create_trainer(net, optimizer, loss_fn, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # forward
        x, y = prepare_batch(batch, device)
        y_pred = net(x)
        loss = loss_fn(y_pred, y)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device)
            y_pred = net(x)
            loss = loss_fn(y_pred, y)
        return x, y_pred, y, loss

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="fc")
    parser.add_argument("--dataset", type=Path, default="data//datasets/foo/RP_dataset/walker_reachability_data_s4_quternion(setp=0.1)_clean.csv")
    parser.add_argument("--logdir", type=Path, default="data/runs/RP")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
