import cv2
import numpy as np

def get_neighbor_by_distance(label_map, neighbor_distance_in_percent=0.02, max_neighbor=32, MAX_INSTANCE=50):
    distance = int(neighbor_distance_in_percent * label_map.shape[0])
    label_map = label_map.copy()
    def _adjust_size(x):
        if len(x) >= max_neighbor:
            return x[0:max_neighbor]
        else:
            return np.pad(x, (0, max_neighbor-len(x)), 'constant',  constant_values=(0, 0))

    unique = np.unique(label_map)
    assert unique[0] == 0
    # only one object
    if len(unique) <= 2:
        return None

    neighbor_indice = np.zeros((len(unique)-1, max_neighbor))
    label_flat = label_map.reshape((-1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance * 2 + 1, distance * 2 + 1))
    for i, label in enumerate(unique[1:]):
        assert i+1 == label
        mask = label_map == label
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).reshape((-1))
        neighbor_pixel_ind = np.logical_and(dilated_mask > 0, label_flat != 0)
        neighbor_pixel_ind = np.logical_and(neighbor_pixel_ind, label_flat != label)
        neighbors = np.unique(label_flat[neighbor_pixel_ind])
        neighbor_indice[i,:] = _adjust_size(neighbors) 

    neighbor_indice = neighbor_indice.astype(np.int32)
    neighbor_indice = np.pad(neighbor_indice, ((0,MAX_INSTANCE-len(unique)+1),(0,0)), mode='constant')
    return neighbor_indice


if __name__ == '__main__':
    # label = cv2.imread(r'D:\expriments\affinity_CVPPP\data\A1\train\plant001_label.png')
    from PIL import Image
    import torch
    import torch.nn.functional as F
    label = np.asarray(Image.open(r'D:\expriments\affinity_CVPPP\data\A1\train\plant001_label.png'))
    print(label.shape)

    MAX_INSTANCE = 32
    neighbor = get_neighbor_by_distance(label)
    # neighbor = np.pad(neighbor, ((0,MAX_INSTANCE-neighbor.shape[0]),(0,0)), mode='constant')
    # print(neighbor)
    print(neighbor.shape)

    # neighbor = torch.from_numpy(neighbor)
    # MAX_INSTANCE = 500
    # padding = ((0, MAX_INSTANCE - neighbor.shape[0]), (0, 0))
    # neighbor = F.pad(neighbor, padding, 'constant', 0)
    # print(neighbor.shape)