import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm import tqdm

from models.data_helper import get_r2_val_dataloader
from models.relational_transformer import RelationalTransformer
from models.resnet import ResNetFc
from utils.ckpt_utils import load_ckpt


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default="ImageNet", choices=["ImageNet"], help="Dataset name")
    parser.add_argument("--path_to_txt", default="data/txt_lists/", help="Path to the txt files")

    # training parameters
    parser.add_argument("--network", default="resnet18", help="Network: resnet18")  # backbone
    parser.add_argument("--n_classes", type=int, default=1, help="Number of outputs for rel head")
    parser.add_argument("--hidden_head_dim", type=int, default=256,
                        help="Dimensionality of the hidden layer in rel transformer head")

    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")

    # relational estimator
    parser.add_argument("--transf_depth", type=int, default=4,
                        help="Number of self attention blocks in relational transformer")
    parser.add_argument("--transf_n_heads", type=int, default=12,
                        help="Number of heads in self attention modules for the relational transformer")

    # checkpoint evaluation
    parser.add_argument("--checkpoint_folder_path", default="outputs/", help="Folder in which the checkpoint is saved")

    # pair sampling
    parser.add_argument("--n_anchors_per_cat", type=int, help="Number of samples for each anchor cat", default=2)
    parser.add_argument("--n_different_cat", type=int, help="Number of other categories to consider to form pairs", default=5)
    parser.add_argument("--n_pos", type=int, help="Number of samples from same cat to consider to form positive pairs", default=1)
    parser.add_argument("--n_neg_per_cat", type=int, help="Number of samples in each other categories to consider to form pairs", default=1)

    args = parser.parse_args()

    args.path_dataset = os.path.expanduser('~/data/')
    print(f"Loading data from: {args.path_dataset}")

    return args


def get_tensor_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )


def cosine_dist(x1, x2):
    return torch.squeeze(1 - F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=2))


def compute_pair_dists(datasets, device, metric_f):
    dists = []
    for l1, l2 in [(0, 0), (1, 1), (0, 1)]:
        loader1 = get_tensor_dataloader(datasets[l1], 512)
        loader2 = get_tensor_dataloader(datasets[l2], 512)
        pair_dists = []
        for (data1,) in tqdm(loader1, ncols=60):
            data1 = data1.to(device)
            for (data2,) in loader2:
                data2 = data2.to(device)
                d = metric_f(data1, data2)
                pair_dists.append(d.view(-1))
        dists.append(torch.cat(pair_dists))
    return dists


def extract_relations(args, device):
    feature_extractor = ResNetFc(device, args.network)
    cls_rel = RelationalTransformer(
        feature_extractor.output_num(),
        num_classes=args.n_classes,
        hidden_head_dim=args.hidden_head_dim,
        depth=args.transf_depth,
        num_heads=args.transf_n_heads,
    )

    feature_extractor.to(device)
    cls_rel.to(device)

    models = {"feature_extractor": feature_extractor, "cls_rel": cls_rel}

    load_ckpt(models, args.checkpoint_folder_path)

    feature_extractor.eval()
    cls_rel.eval()

    loader = get_r2_val_dataloader(args)

    relations = []
    relation_ls = []

    print("Extracting relations")

    for batch in tqdm(loader, ncols=60):
        data1, data2, _, relation_l = batch
        data1, data2 = data1.to(device), data2.to(device)

        # forward
        batch_size = data1.shape[0]
        data_tot = torch.cat((data1, data2))
        data_tot = feature_extractor(data_tot)
        data1_feat = data_tot[:batch_size]
        data2_feat = data_tot[-batch_size:]

        data12_aggregation = torch.cat((data1_feat, data2_feat), 1)

        # compute relation
        _, feats, head_feats = cls_rel(
            data12_aggregation, return_feats=True, return_head_feats=True
        )

        # take intermediate feats only if head is a MLP
        relation = feats if args.hidden_head_dim is None else head_feats

        relations.append(relation.cpu())
        relation_ls.append(relation_l.cpu())

    relations = torch.cat(relations)
    relation_ls = torch.cat(relation_ls)

    return relations, relation_ls


@torch.no_grad()
def r2_eval(args, device):
    lbls = (0, 1)

    relations, relation_ls = extract_relations(args, device)

    datasets = tuple(TensorDataset(relations[relation_ls == l]) for l in lbls)

    print("Computing pair distances")
    dists = compute_pair_dists(datasets, device, cosine_dist)

    # compute R2
    avg_dists = [torch.mean(d) for d in dists]
    avg_dists.append(avg_dists[-1])   # (1, 0) is equal to (0, 1)
    avg_dist_total = sum(avg_dists) / (len(lbls) ** 2)
    avg_dist_within = sum(avg_dists[:2]) / len(lbls)
    r2 = 1 - (avg_dist_within / avg_dist_total)

    return r2
    

def main():
    args = get_args()

    print(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("WARNING. Running in CPU mode")
        device = torch.device("cpu")

    assert os.path.isdir(
        f"{args.checkpoint_folder_path}"
    ), "Cannot perform visualization! Checkpoint path does not exist!"
    
    r2 = r2_eval(args, device)
    print(f"R2: {r2}")


if __name__ == "__main__":
    main()
