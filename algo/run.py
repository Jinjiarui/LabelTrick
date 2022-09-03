#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import sys
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl import function as fn
from dgl.data import (
    AmazonCoBuyComputerDataset,
    AmazonCoBuyPhotoDataset,
    CiteseerGraphDataset,
    CoauthorCSDataset,
    CoraFullDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    RedditDataset,
)
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from tqdm import tqdm

from models import GAT, GCN, MLP, LabelPropagation

epsilon = 1 - math.log(2)

device = None

n_node_feats, n_edge_feats, n_classes = 0, 0, 0


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def compute_acc(pred, labels):
    return ((torch.argmax(pred, dim=1) == labels[:, 0]).float().sum() / len(pred)).item()


def load_data(dataset, split):
    global n_node_feats, n_classes

    if dataset in ["ogbn-arxiv", "ogbn-proteins", "ogbn-products"]:
        data = DglNodePropPredDataset(name=dataset)
    elif dataset == "cora":
        data = CoraGraphDataset()
    elif dataset == "citeseer":
        data = CiteseerGraphDataset()
    elif dataset == "pubmed":
        data = PubmedGraphDataset()
    elif dataset == "cora-full":
        data = CoraFullDataset()
    elif dataset == "reddit":
        data = RedditDataset()
    elif dataset == "amazon-co-computer":
        data = AmazonCoBuyComputerDataset()
    elif dataset == "amazon-co-photo":
        data = AmazonCoBuyPhotoDataset()
    elif dataset == "coauthor-cs":
        data = CoauthorCSDataset()
    else:
        assert False

    if dataset in ["ogbn-arxiv", "ogbn-products"]:
        graph, labels = data[0]
        graph = graph.to_simple(graph)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )

        evaluator_ = Evaluator(name=dataset)
        evaluator = lambda pred, labels: evaluator_.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
        )["acc"]

        if dataset == "ogbn-proteins":
            graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))

    elif dataset in [
        "cora",
        "citeseer",
        "pubmed",
        "reddit",
    ]:
        graph = data[0]
        labels = graph.ndata["label"].reshape(-1, 1)
        train_mask, val_mask, test_mask = (
            graph.ndata["train_mask"],
            graph.ndata["val_mask"],
            graph.ndata["test_mask"],
        )
        train_idx, val_idx, test_idx = map(
            lambda mask: torch.nonzero(mask, as_tuple=False).squeeze_(), [train_mask, val_mask, test_mask],
        )

        evaluator = compute_acc

    elif dataset in ["cora-full", "amazon-co-computer", "amazon-co-photo", "coauthor-cs"]:
        graph = data[0]
        graph = graph.to_simple(graph)
        labels = graph.ndata["label"].reshape(-1, 1)
        train_idx, val_idx, test_idx = None, None, None
        assert split == "random"

        evaluator = compute_acc
    else:
        assert False

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    print(f"#Nodes: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}, #Classes: {n_classes}")
    if split != "random":
        print(f"#Train/Val/Test nodes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def propagate(graph, feat, prop_step=8, alpha=0.1):
    graph.ndata["h"] = feat

    norm = torch.pow(graph.out_degrees().float().clamp(min=1), 0.5).reshape(-1, 1)
    for _ in range(prop_step):
        graph.ndata["h"] = graph.ndata["h"] / norm
        graph.update_all(fn.copy_src(src="h", out="m"), fn.mean(msg="m", out="h"))
        graph.ndata["h"] = graph.ndata["h"] * norm
        graph.ndata["h"] = alpha * graph.ndata["h"] + (1 - alpha) * feat

    return graph.ndata["h"]


def diag_p(graph, target_idx, prop_step=7, alpha=0.1, block_size=1024):
    if "diag_p" in graph.ndata:
        return graph.ndata["diag_p"]

    res = torch.zeros(graph.number_of_nodes(), 1, device=graph.device)
    for i in tqdm(range(0, target_idx.shape[0], block_size)):
        label_idx_batch = target_idx[i : i + block_size]
        n_batch = len(label_idx_batch)

        h = torch.zeros(graph.number_of_nodes(), n_batch, device=device)
        h[label_idx_batch, torch.arange(n_batch)] = 1

        h = propagate(graph, h, prop_step=prop_step, alpha=alpha)
        h = torch.diag(h[label_idx_batch]).reshape(-1, 1)
        res[label_idx_batch] = h

    graph.ndata["diag_p"] = res

    return res


def self_excluded_label_propagation(graph, labels, label_idx):
    label_onehot = torch.zeros([graph.number_of_nodes(), n_classes], device=device)
    label_onehot[label_idx, labels[label_idx, 0]] = 1

    label_prop = propagate(graph, label_onehot)
    label_coeff = diag_p(graph, label_idx)
    label_prop -= label_coeff * label_onehot

    return label_prop


def preprocess(args, graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    if args.no_feat:
        graph.ndata["feat"] = torch.zeros(graph.number_of_nodes(), 0)

    graph.create_formats_()

    return graph


def random_split(graph):
    n = graph.number_of_nodes()
    perm = torch.randperm(n, device=device)
    val_offset, test_offset = int(n * 0.6), int(n * 0.8)
    train_idx, val_idx, test_idx = (
        perm[:val_offset],
        perm[val_offset:test_offset],
        perm[test_offset:],
    )

    print(f"#Train/Val/Test nodes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    return train_idx, val_idx, test_idx


def build_model(args):
    n_input_feats = 0
    if not args.no_feat:
        n_input_feats += n_node_feats
    if args.labels:
        n_input_feats += n_classes

    if args.activation == "relu":
        activation = F.relu
    elif args.activation == "elu":
        activation = F.elu
    elif args.activation == "none":
        activation = None
    else:
        assert False

    if args.model == "mlp":
        model = MLP(
            in_feats=n_input_feats,
            n_hidden=args.n_hidden,
            n_classes=n_classes,
            n_layers=args.n_layers,
            activation=activation,
            norm=args.norm,
            input_drop=args.input_drop,
            dropout=args.dropout,
            residual=args.residual,
        )
    elif args.model == "gcn":
        model = GCN(
            in_feats=n_input_feats,
            n_classes=n_classes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            activation=activation,
            norm=args.norm,
            norm_adj=args.norm_adj,
            input_drop=args.input_drop,
            dropout=args.dropout,
            # use_linear=args.linear,
            # residual=args.residual,
        )
    elif args.model == "gat":
        model = GAT(
            dim_node=n_input_feats,
            dim_edge=n_edge_feats,
            dim_output=n_classes,
            n_hidden=args.n_hidden,
            edge_emb=16,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=activation,
            norm=args.norm,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            edge_drop=args.edge_drop,
            non_interactive_attn=args.non_interactive_attn,
            # negative_slope=args.negative_slope,
            use_symmetric_norm=args.norm_adj == "symm",
            linear=args.linear,
            residual=args.residual,
        )
    elif args.model == "lp":
        # exit(0)
        model = LabelPropagation(
            channels=n_input_feats,
            n_prop=args.n_prop,
            alpha=args.alpha,
            param=args.param,
            norm=args.norm_adj,
            clamp=args.clamp,
            fixed_input=not (args.labels and args.mask_rate < 1),
            fixed_feat=True,
        )
    else:
        assert False

    return model


def compute_loss(args, x, labels):
    if args.loss == "mse":
        x = x[:, :n_classes]
        onehot = torch.zeros([labels.shape[0], n_classes], device=device)
        onehot[:, labels[:, 0]] = 1
        return torch.mean((x - onehot) ** 2)

    x = torch.softmax(x, dim=-1)
    y = -torch.log(1e-6 + x[range(x.shape[0]), labels[:, 0]])
    if args.loss == "loge":
        y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx, scale=1):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = scale
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def general_outcome_correlation(graph, y0, n_prop=7, alpha=0.8, use_norm=False, post_step=None):
    with graph.local_scope():
        y = y0
        for _ in range(n_prop):

            graph.srcdata.update({"y": y})
            graph.update_all(fn.copy_u("y", "m"), fn.mean("m", "y"))
            y = graph.dstdata["y"]

            y = alpha * y + (1 - alpha) * y0

            if post_step is not None:
                y = post_step(y)

        return y


def correct_and_smooth(args, graph, y, labels, train_labels_idx):
    y = y.clone()
    y[train_labels_idx] = F.one_hot(labels[train_labels_idx], n_classes).float().squeeze(1)

    smoothed_y = general_outcome_correlation(
        graph, y, alpha=args.alpha, use_norm=args.norm_adj == "symm",  # , post_step=lambda x: x.clamp(0, 1)
    )

    return smoothed_y


def train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, evaluator):
    model.train()
    torch.autograd.set_detect_anomaly(True)

    feat = graph.ndata["feat"]

    c = None

    if args.labels:
        if args.mask_rate == 1:
            c = diag_p(graph, train_idx, prop_step=args.n_prop, alpha=args.alpha)
            feat = add_labels(feat, labels, train_idx)
            train_labels_idx = train_idx
            train_pred_idx = train_idx
        else:
            mask = torch.rand(train_idx.shape) < args.mask_rate

            train_labels_idx = train_idx[mask]
            train_pred_idx = train_idx[~mask]

            feat = add_labels(feat, labels, train_labels_idx, scale=1 / args.mask_rate)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_pred_idx = train_idx[mask]

    if args.model == "mlp":
        y = model(feat)
    elif args.model == "lp":
        y = model(graph, feat, c=c, n_classes=n_classes)
    else:
        y = model(graph, feat)

    loss = compute_loss(args, y[train_pred_idx], labels[train_pred_idx])

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return evaluator(y[train_pred_idx], labels[train_pred_idx]), loss.item()


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, evaluator, epoch):
    model.eval()

    c = None
    feat = graph.ndata["feat"]

    if args.labels:
        feat = add_labels(feat, labels, train_idx)

    if args.model == "mlp":
        y = model(feat)
    elif args.model == "lp":
        y = model(graph, feat, c=c, n_classes=n_classes)
    else:
        y = model(graph, feat)

    train_loss = compute_loss(args, y[train_idx], labels[train_idx])
    val_loss = compute_loss(args, y[val_idx], labels[val_idx])
    test_loss = compute_loss(args, y[test_idx], labels[test_idx])

    return (
        evaluator(y[train_idx], labels[train_idx]),
        evaluator(y[val_idx], labels[val_idx]),
        evaluator(y[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        y,
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    if args.split == "random":
        train_idx, val_idx, test_idx = random_split(graph)

    # define model and optimizer
    model = build_model(args).to(device)

    if args.optimizer == "none":
        optimizer = None
    elif args.model == "lp" and not args.param:
        optimizer = None
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.8)
    else:
        assert False

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
    final_pred = None
    best_epoch = 0

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        tic = time.time()

        if args.optimizer == "rmsprop":
            adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss = train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer, evaluator,)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, labels, train_idx, val_idx, test_idx, evaluator, epoch
        )

        toc = time.time()
        total_time += toc - tic

        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_pred = pred
            best_epoch = epoch

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses,],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

        if epoch == args.epochs or epoch % args.log_every == 0:
            print(
                f"Run: {n_running}/{args.runs}, Epoch: {epoch}/{args.epochs}, Average epoch time: {total_time / epoch:.4f}s, Best epoch: {best_epoch}\n"
                f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )
            if epoch - best_epoch > 500:
                break

    print("*" * 50)
    print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    print("*" * 50)

    # plot learning curves
    if args.plot:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([accs, train_accs, val_accs, test_accs], ["acc", "train acc", "val acc", "test acc"],):
            plt.plot(range(args.epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.model}_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"],
        ):
            plt.plot(range(args.epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.model}_loss_{n_running}.png")

    if args.save_pred:
        os.makedirs("./output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{n_running}.pt")

    return best_val_acc, final_test_acc


def count_parameters(args):
    model = build_model(args)
    return sum([p.numel() for p in model.parameters()])


def main():
    global device

    argparser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    # basic settings
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument("--runs", type=int, default=10, help="running times")
    argparser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "ogbn-arxiv",
            "ogbn-proteins",
            "ogbn-products",
            "cora",
            "citeseer",
            "pubmed",
            "cora-full",
            "reddit",
            "amazon-co-computer",
            "amazon-co-photo",
            "coauthor-cs",
        ],
        default="ogbn-arxiv",
        help="dataset",
    )
    argparser.add_argument("--split", type=str, choices=["std", "random"], default="std", help="split")
    # training
    argparser.add_argument("--epochs", type=int, default=2000, help="number of epochs")
    argparser.add_argument(
        "--loss", type=str, choices=["mse", "logit", "loge", "savage"], default="logit", help="loss function",
    )
    argparser.add_argument(
        "--optimizer", type=str, choices=["none", "adam", "rmsprop", "sgd"], default="adam", help="optimizer",
    )
    argparser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    # model
    argparser.add_argument("--no-feat", action="store_true", help="do not use node features")
    argparser.add_argument(
        "--labels", action="store_true", help="use labels in the training set as input features",
    )
    argparser.add_argument("--n-label-iters", type=int, default=0, help="number of label iterations")
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument(
        "--model", type=str, choices=["mlp", "gcn", "gat", "lp", "twirls"], default="gat", help="model",
    )
    argparser.add_argument("--residual", action="store_true", help="residual")
    argparser.add_argument("--linear", action="store_true", help="use linear layer")
    argparser.add_argument(
        "--norm-adj",
        type=str,
        choices=["symm", "rw", "default", "ad"],
        default="default",
        help="symmetric normalized (symm) or randon walk normalized (rw) adjacency matrix; default for GCN: symm, default for GAT: rw",
    )
    argparser.add_argument("--non-interactive-attn", action="store_true", help="non-interactive attention")
    argparser.add_argument("--norm", type=str, choices=["none", "batch"], default="batch", help="norm")
    argparser.add_argument(
        "--activation", type=str, choices=["none", "relu", "elu"], default="relu", help="activation",
    )
    argparser.add_argument("--n-prop", type=int, default=7, help="number of props")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--n-hidden", type=int, default=256, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.0, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention drop rate")
    argparser.add_argument("--edge-drop", type=float, default=0.0, help="edge drop rate")
    argparser.add_argument("--alpha", type=float, default=0.6, help="edge drop rate")
    argparser.add_argument("--param", action="store_true", help="param")
    argparser.add_argument("--clamp", action="store_true", help="clamp")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    # output
    argparser.add_argument("--log-every", type=int, default=20, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--tune", type=str, default="", help="tune")
    args = argparser.parse_args()

    if not args.labels and args.n_label_iters > 0:
        raise ValueError("'--labels' must be enabled when n_label_iters > 0")

    if args.model == "gcn":
        if args.non_interactive_attn > 0:
            raise ValueError("'no_attn_dst' is not supported for GCN")
        if args.attn_drop > 0:
            raise ValueError("'attn_drop' is not supported for GCN")
        if args.edge_drop > 0:
            raise ValueError("'edge_drop' is not supported for GCN")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    if args.norm_adj == "default":
        if args.model == "gcn":
            args.norm_adj = "symm"
        elif args.model == "gat":
            args.norm_adj = "rw"
        else:
            args.norm_adj = "symm"

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(args.dataset, args.split)
    graph = preprocess(args, graph)

    graph, labels = map(lambda x: x.to(device), (graph, labels))
    if args.split != "random":
        train_idx, val_idx, test_idx = map(lambda x: x.to(device), (train_idx, val_idx, test_idx))

    # run
    val_accs, test_accs = [], []

    for i in range(args.runs):
        seed(args.seed + i)
        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i + 1)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    # print results
    print(" ".join(sys.argv))
    print(args)
    if args.runs > 0:
        print(f"Runned {args.runs} times")
        print("Val Accs:", val_accs)
        print("Test Accs:", test_accs)
        print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
        print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"#Params: {count_parameters(args)}")


if __name__ == "__main__":
    main()
