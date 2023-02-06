from numpy.core.numeric import Inf
import open3d as o3d
import numpy as np
import statistics
from queue import SimpleQueue
from copy import deepcopy
from threading import Thread
from time import time

# Python implementation of the algorithm proposed in:
#   Lin Y, Wang C, Zhai D, W Li, and J Li. Toward better boundary preserved supervoxel segmentation for 3D point clouds. 
#   ISPRS Journal of Photogrammetry & Remote Sensing, vol. 143, pages 39-47, 2018.
#
# The authors published some C++ code, that was very helpful to produce this implementation, here:
#   https://github.com/yblin/Supervoxel-for-3D-point-clouds
#

class Supervoxels(Thread):
    def __init__(self, open3d_point_cloud, desired_nb_supervoxels=24, K_NN=10):
        Thread.__init__(self, group=None, target=None, name=None, args=None, kwargs=None, daemon=True)
        #Point-cloud built with open3d.geometry.PointCloud() 
        self.pcd = open3d_point_cloud
        #Number of supervoxels we want to obtain (depends on object shape)
        self.desired_nb_supervoxels = desired_nb_supervoxels
        #Size of neighborhood (depends on point-cloud density)
        self.K_NN = K_NN
        #Greatest extent of the point-cloud in meter
        bb_extent = self.pcd.get_oriented_bounding_box().extent
        bb_diag   = np.sqrt(bb_extent[0]**2 + bb_extent[1]**2 + bb_extent[2]**2)
        self.max_extent = bb_diag

    #Compute similarity metric between point i and point j
    def similarity_metric(self, points, colors, normals, i, j):
        c1 = 10
        c2 = 1
        c3 = 1

        dist_position = c1*np.linalg.norm(points[i]-points[j])       #0.126 second
        dist_colors   = c2*np.linalg.norm(colors[i]-colors[j])       #0.1232 second
        dist_normals  = c3*(1-np.abs(np.dot(normals[i],normals[j]))) #0.117 second

        dissimilarity = dist_position + dist_colors + dist_normals

        return dissimilarity

    #Finds the supervoxel of the i-th point
    def find_supervoxel(self, list_sets, i):
        #Heuristic to make it faster, since at the beginning, the i-th point is in the i-th subset
        if i in list_sets[i]:
            return i

        for k, subset in enumerate(list_sets):
            if i in subset:
                return k
        raise IndexError('Cannot find point in any subset.')

    #Called when thread is done
    def join(self):
        Thread.join(self)
        return self.supervoxels, self.label

    def run(self):
        sv_start_timer = time()
        points = self.pcd.points
        colors = self.pcd.colors
        normals= self.pcd.normals
        #List of subsets of the points. Each set specify a region/part/segment.
        # The index of the set in the list is the label of the segment.
        # Each subset contains indices of points from the points array such that we dont
        # have to deal with 3D points or complex objects.
        # The list of sets is initialized such that the N-th set contains only point N
        list_sets = [{i} for i in range(0,len(points))]
        #For each subset, contains the representative points only
        representative_points = [{i} for i in range(0,len(points))]

        #Build a KD-tree from the point-cloud
        # Example: Finds the indices (in idx) of the 10 nearest neighbors around the 500-th point
        # [k, idx, _] = kd_tree.search_knn_vector_3d(points[500], 10)
        kd_tree = o3d.geometry.KDTreeFlann(self.pcd)

        #Initialize the region associated with each point using the K-NN
        # and compute the similarity between each point and all its neighbors
        # and keep the lowest similarity/distance.
        neighbors  = []
        region     = []
        similarity = []
        for i,point in enumerate(points):
            [k, idx, _] = kd_tree.search_knn_vector_3d(point, self.K_NN)
            neighbors.append(idx)
            region.append(idx)

            #This block of code performs a large number of objective function evaluations and can be very expensive.
            #Because we use these distances for statistical purposes, we dont need to compute all of them to get a good idea
            # and about 1000 should be enough.
            N_SAMPLES = 1000
            if i % int((len(points)*self.K_NN)/N_SAMPLES) == 0:
                smallest_s = Inf
                for neighbor in idx:
                    if neighbor != i:
                        s = self.similarity_metric(points, colors, normals, i, neighbor)
                        if s < smallest_s:
                            smallest_s = s
                similarity.append(smallest_s)

        #Initialize regularization factor lambda to the median of the similarities
        reg_lambda = statistics.median(similarity)

        #List of supervoxels. At the beginning, every point is a supervoxel.
        supervoxels = [i for i in range(0,len(points))]

        #Perform region growing in this loop
        previous_nb_supervoxels = Inf
        while len(supervoxels) > self.desired_nb_supervoxels:
            previous_nb_supervoxels = len(supervoxels)
            #Consider merging nearby supervoxels in this loop
            for supervoxel in deepcopy(supervoxels):
                #If the supervoxel has not already been merged, consider it.
                if len(region[supervoxel]) > 0:

                    #Supervoxels to consider
                    supervoxels_queue = SimpleQueue()
                    
                    #List of supervoxels that has been considered for merging
                    already_queued_supervoxels = []

                    #Build a list of nearby supervoxels
                    for neighbor in region[supervoxel]:
                        #Find the supervoxel of the neighbor
                        neighbors_supervoxel = self.find_supervoxel(list_sets, neighbor)
                        #Potentially add it to the queue to be considered
                        if neighbors_supervoxel not in already_queued_supervoxels:
                            supervoxels_queue.put(neighbors_supervoxel)
                            already_queued_supervoxels.append(neighbors_supervoxel)

                    #Grow a region
                    new_region = []
                    while not supervoxels_queue.empty():
                        #Return and remove next one
                        considered_supervoxel = supervoxels_queue.get()

                        if considered_supervoxel != supervoxel:
                            #Maximal total dissimilarity
                            # If the metric respects the triangle inequality, the true loss is
                            # guaranteed to be lower than this.
                            c = len(representative_points[considered_supervoxel])
                            sim = self.similarity_metric(points, colors, normals, supervoxel, considered_supervoxel)
                            max_loss = c * sim

                            #If merging would reduce the objective function, do it.
                            if reg_lambda - max_loss > 0:
                                #Merge considered_supervoxel into supervoxel
                                list_sets[supervoxel] = list_sets[supervoxel].union(region[considered_supervoxel]).union(list_sets[considered_supervoxel])
                                representative_points[supervoxel] = representative_points[supervoxel].union(representative_points[considered_supervoxel])
                                #Add merged points into the queue
                                merged_points = set(region[considered_supervoxel]) - set(region[supervoxel])
                                for merged_point in merged_points:
                                    if merged_point not in already_queued_supervoxels:
                                        supervoxels_queue.put(merged_point)
                                        already_queued_supervoxels.append(merged_point)
                                #Clear the region associated with the merged supervoxel
                                region[considered_supervoxel] = []
                                list_sets[considered_supervoxel] = {}
                                representative_points[considered_supervoxel] = {}
                                #Remove considered_supervoxel from supervoxels list
                                if considered_supervoxel in supervoxels:
                                    supervoxels.remove(considered_supervoxel)
                            else:
                                new_region.append(considered_supervoxel)
                    region[supervoxel] = new_region 
                    if len(supervoxels) <= self.desired_nb_supervoxels:
                        break

            #At the end, increase lambda to converge.
            #LAMBDA_FACTOR must be higher than 1, a smaller LAMBDA_FACTOR seems to improve segmentation at the expense of time
            #   LAMBDA_FACTOR = 2.00 : 0.8 seconds
            #   LAMBDA_FACTOR = 1.50 : 1.07 seconds
            #   LAMBDA_FACTOR = 1.10 : 2.7 seconds
            #   LAMBDA_FACTOR = 1.01 : 20.4 seconds
            LAMBDA_FACTOR = 2
            reg_lambda = reg_lambda * LAMBDA_FACTOR

        #Each point is labeled using its associated supervoxel
        # and the similarity between the two is computed.
        label = []
        similarity_to_supervoxel = []
        for i in range(0,len(points)):
            associated_supervoxel = self.find_supervoxel(list_sets, i)
            label.append(associated_supervoxel)
            sim = self.similarity_metric(points, colors, normals, i, associated_supervoxel)
            similarity_to_supervoxel.append(sim)

        #For all points, add to the queue its neighbors that dont share the same label. 
        # This has the effect of adding to the queue only the points that are on the border of the segments.
        # We use a set() in order to avoid duplicate elements
        points_queue = set()
        for point in range(0,len(points)):
            for i,neighbor in enumerate(neighbors[point]):
                if label[point] != label[neighbor]:
                        points_queue.add(point)
                        points_queue.add(neighbor)

        #Refine the boundaries of the regions by looking at the borders
        while len(points_queue) > 0:
            selected_point = points_queue.pop()
            #If the selected point is a supervoxel, it cannot be relabeled
            if label[selected_point] != selected_point:
                for neighbor in neighbors[selected_point]:
                    if label[neighbor] != label[selected_point]:
                        #Compute the similarity between the selected point and the neighbor's supervoxel
                        sim = self.similarity_metric(points, colors, normals, selected_point, label[neighbor])
                        #If the selected point is more similar to the neighbor's supervoxel 
                        # than its currently associated supervoxel
                        if sim < similarity_to_supervoxel[selected_point]:
                            #Associate the point with the neighbor's supervoxel
                            label[selected_point] = label[neighbor]
                            similarity_to_supervoxel[selected_point] = sim
                            #Add the neighbors of the selected point to the queue
                            # as they are probably now part of the border
                            for n in neighbors[selected_point]:
                                points_queue.add(n)

        #Relabel the supervoxels
        labels_mapping = {supervoxels[i]: i for i in range(0, len(supervoxels))}
        #Reset the labels of every point
        for point in range(0, len(points)):
            old_label    = label[point]
            if old_label not in supervoxels:
                raise ValueError('The label of a point is not considered a supervoxel.')
            new_label    = labels_mapping[old_label]
            label[point] = new_label
        
        #Variables to be returned
        self.supervoxels = supervoxels
        self.label = label

        print("Producing supervoxels took {} seconds".format(time()-sv_start_timer))
