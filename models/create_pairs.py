import random
import os
import numpy as np
from tqdm import tqdm

def preprocess_dataset(files):
    lines = [f.strip() for f in files]
    files = []
    lbls = []
    for l in lines:
        file_path, lbl = l.split()
        files.append(file_path)
        lbls.append(int(lbl))

    lbl_set = set(lbls)
    indices = np.arange(len(lbls))
    lbls = np.array(lbls)

    cat_indices = {}
    for lbl in lbl_set:
        cat_indices[lbl] = indices[lbls==lbl]
    
    return files, lbls, cat_indices, lbl_set


def create_pairs_source_v2(path_to_txt, enable_class_balancing=True, neg_to_pos_ratio=1):

    print("Generate source pairs")
    pairs = []

    # for each category we count occurrences
    cat_count = {}

    with open(path_to_txt) as input_file:
        file_names = input_file.readlines()

    files_paths, lbls, cat_indices, lbl_set = preprocess_dataset(file_names)

    max_cardinality = 0
    tuple_lbl_set = tuple(lbl_set)
    for k, v in cat_indices.items():
        cat_count[k] = len(v)
        if len(v) > max_cardinality:
            max_cardinality = len(v)

    for list_idx in tqdm(range(len(files_paths))):
        file_1, category = files_paths[list_idx], lbls[list_idx]

        # other file same category
        idx = np.random.choice(cat_indices[category])
        other_file = files_paths[idx]
        pairs.append(file_1+' '+other_file+' '+str(category)+' '+str(0)+'\n')

        # other file different category
        for neg in range(neg_to_pos_ratio):

            different_cat = category
            while different_cat == category:
                different_cat = random.choice(tuple_lbl_set)
            idx = np.random.choice(cat_indices[different_cat])

            other_file = files_paths[idx]
            pairs.append(file_1+' '+other_file+' '+str(category)+' '+str(1)+'\n')

    if enable_class_balancing:
        print("Generate additional pairs for class balancing")
        for lbl in tqdm(range(len(cat_count))):
            while cat_count[lbl] < max_cardinality:
                idx_1 = np.random.choice(cat_indices[lbl])
                file_1 = files_paths[idx_1]
                idx_2 = np.random.choice(cat_indices[lbl])
                file_2 = files_paths[idx_2]

                pairs.append(file_1+' '+file_2+' '+str(lbl)+' '+str(0)+'\n')

                for neg in range(neg_to_pos_ratio):
                    different_cat = lbl
                    while different_cat == lbl:
                        different_cat = random.choice(tuple_lbl_set)
                    idx_2 = np.random.choice(cat_indices[different_cat])

                    file_2 = files_paths[idx_2]
                    pairs.append(file_1+' '+file_2+' '+str(lbl)+' '+str(1)+'\n')

                cat_count[lbl] += 1
    return pairs


def create_pairs_r2_val(path_to_txt, n_anchors_per_cat=None, n_different_cat=None, n_pos=None, n_neg_per_cat=None):
    """Create pairs for binary class evaluation

    Args:
        path_to_txt: the path to the txt files
        n_anchors_per_cat: number of anchor files to consider for each category (the first elements in the pairs)
        n_different_cat: the number of different categories to consider to form the negative pairs
        n_pos: the number of positive pairs to build
        n_neg_per_cat: the number of negatives to build for each different category
    """

    same_lbl, diff_lbl = 0, 1

    print("Generate source pairs")
    pairs = []

    with open(path_to_txt) as input_file:
        file_names = input_file.readlines()

    files_paths, _, cat_indices, lbl_set = preprocess_dataset(file_names)

    # Iterate over all categories
    for anchor_cat in tqdm(lbl_set, ncols=60):

        # pick n_anchors_per_cat random files (used as "anchors") from the considered category
        anchor_cat_indices = (
            np.random.choice(cat_indices[anchor_cat], n_anchors_per_cat, replace=False)
            if n_anchors_per_cat is not None and n_anchors_per_cat < len(cat_indices[anchor_cat]) else
            cat_indices[anchor_cat]
        )

        # Iterate over the chosen random files
        for file_idx in anchor_cat_indices:
            file_1 = files_paths[file_idx]

            ### Positive pairs ###

            # select the indices from the same cat to form the positive pairs
            other_pos_indices = cat_indices[anchor_cat][np.where(cat_indices[anchor_cat] != file_idx)]
            if n_pos is not None and n_pos < len(other_pos_indices):
                other_pos_indices = np.random.choice(other_pos_indices, n_pos, replace=False)

            # form the positive pairs
            for other_idx in other_pos_indices:
                other_file = files_paths[other_idx]
                pairs.append(file_1+' '+other_file+' '+str(anchor_cat)+' '+str(same_lbl)+'\n')

            ### Negative pairs ###

            # consider only a subset of "other" categories
            different_cats = [c for c in lbl_set if c != anchor_cat]
            if n_different_cat is not None and n_different_cat < len(different_cats):
                different_cats = random.sample(different_cats, n_different_cat)

            other_neg_indices = []

            for different_cat in different_cats:
                if different_cat == anchor_cat:
                    continue
                
                different_cat_indices = cat_indices[different_cat]
                if n_neg_per_cat is not None and n_neg_per_cat < len(different_cat_indices):
                    different_cat_indices = np.random.choice(different_cat_indices, n_neg_per_cat, replace=False)

                other_neg_indices.append(different_cat_indices)

            other_neg_indices = np.concatenate(other_neg_indices) 

            for other_idx in other_neg_indices:
                other_file = files_paths[other_idx]
                pairs.append(file_1+' '+other_file+' '+str(anchor_cat)+' '+str(diff_lbl)+'\n')

    return pairs
