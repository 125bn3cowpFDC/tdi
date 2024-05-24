import numpy as np
from . import tools

# Joint index:
# {0,  "Nose"}  -->  0
# {1,  "Neck"}  -->  17,
# {2,  "RShoulder"}  -->  5,
# {3,  "RElbow"}  -->  7,
# {4,  "RWrist"}  -->  9,
# {5,  "LShoulder"}  -->  6,
# {6,  "LElbow"}  -->  8,
# {7,  "LWrist"}  -->  10,
# {8,  "RHip"}  -->  11,
# {9,  "RKnee"}  -->  13,
# {10, "RAnkle"}  -->  15,
# {11, "LHip"}  -->  12,
# {12, "LKnee"}  -->  14,
# {13, "LAnkle"}  -->  16,
# {14, "REye"}  -->  1,
# {15, "LEye"}  -->  2,
# {16, "REar"}  -->  3,
# {17, "LEar"}  -->  4,

# Edge format: (origin, neighbor)
num_node = 18
self_link = [(i, i) for i in range(num_node)]
'''
# ORIGINAL INFORMATION
inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14)]
'''
inward = [(9, 7), (7, 5), (10, 8), (8, 6), (16, 14), (14, 12), (15, 13), (13, 11),
          (12, 6), (11, 5), (6, 17), (5, 17), (0, 17), (2, 0), (1, 0), (4, 2),
          (3, 1)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix

    For more information, please refer to the section 'Partition Strategies' in our paper.

    """

    def __init__(self, labeling_mode='uniform'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'DAD':
            A = tools.get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = tools.get_DLD_graph(num_node, self_link, neighbor)
        # elif labeling_mode == 'customer_mode':
        #     pass
        else:
            raise ValueError()
        return A


def main():
    mode = ['uniform', 'distance*', 'distance', 'spatial', 'DAD', 'DLD']
    np.set_printoptions(threshold=np.nan)
    for m in mode:
        print('=' * 10 + m + '=' * 10)
        print(Graph(m).get_adjacency_matrix())


if __name__ == '__main__':
    main()
