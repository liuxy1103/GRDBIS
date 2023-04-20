import numpy as np
import elf.segmentation.multicut as mc
import elf.segmentation.features as feats
import elf.segmentation.watershed as ws

def multicut_multi(affs, offsets=[[-1, 0], [0, -1]], fragments=None):
    affs = 1 - affs
    import pdb
    pdb.set_trace()
    boundary_input = np.maximum(affs[0], affs[1])
    fragments, _ = ws.distance_transform_watershed(boundary_input, threshold=.25, sigma_seeds=2.)
    rag = feats.compute_rag(fragments)
    costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    return segmentation
