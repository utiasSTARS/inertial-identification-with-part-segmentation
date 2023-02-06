#This is an implementation of the algorithm described in the following paper:
# @inproceedings{attene2008hierarchical,
#   title={Hierarchical convex approximation of 3D shapes for fast region selection},
#   author={Attene, Marco and Mortara, Michela and Spagnuolo, Michela and Falcidieno, Bianca},
#   booktitle={Computer graphics forum},
#   volume={27},
#   number={5},
#   pages={1323--1332},
#   year={2008},
#   organization={Wiley Online Library}
# }
#
# As far as I know, there is no other public implementation available.
import numpy as np
import pydot
from sortedcontainers import SortedKeyList
from time import time
from scipy.spatial import ConvexHull
from pathlib import Path
from EfficientTetrahedronSearch import TetSearch
from VolumeReconstruction import TetraMeshFromPointCloud

class HierarchicalTetClustering(object):
    def __init__(self,
        points: np.ndarray, 
        normals: np.ndarray, 
        tetgen_executable_path: Path, 
        profiling: bool = False) -> None:

        #Whether we want to print timing information or not
        self.profiling = profiling

        #Generate the tetrahedral mesh
        tetragen = TetraMeshFromPointCloud(points, normals, tetgen_executable_path, profile_execution=profiling)
        self.tet_points, self.tet_list = tetragen.generate()

        if profiling:
            start_time = time()

        #Setup the efficient tettrahedron search
        self.tet_search = TetSearch(self.tet_list)
        if profiling:
            print("Faces sorted in {} seconds.".format(time()-start_time))
            start_time = time()

        #Assumes that tet_list is a list of 4D int vectors that defines the tetrahedra
        # each int being the index of a point in tet_points
        self.nb_tet = len(self.tet_list)
        #Volume of the n-th cluster
        self.cluster_volume = {}
        #A list of parent clusters such that children[parent]=[Lcluster, Rcluster]
        self.children = []
        #A list of parent clusters such that parent[children]=parent
        self.parent   = {}
        #A list that maps each edge to its associated cost
        self.costs    = []
        #A list of clusters/'dual vertice' in the 'dual graph'; each clusters contains indices of tet
        self.clusters = []
        #A list of 'dual edges' in the 'dual graph'; each edge contains the indices of two clusters
        self.edges    = []
        #A sorted key list of edge indices; a pop operation retrieve the index of the edge with the lowest cost
        self.edge_indices = SortedKeyList([], key=self.cost_key)
        #A list of dual edges that are connected to a dual vertex
        self.vertex_edges = [[] for t in self.tet_list]

    #Returns the volume of the n-th tet
    def tet_volume(self, n):
        v1, v2, v3, v4 = self.tet_list[n]
        return self.tet_volume_from_points(v1, v2, v3, v4)

    #Returns the volume of tet defined by the 4 vertices give as arguments
    def tet_volume_from_points(self, v1, v2, v3, v4, signed=False):
        t1 = self.tet_points[v1]
        t2_wrt_t1 = self.tet_points[v2] - t1
        t3_wrt_t1 = self.tet_points[v3] - t1
        t4_wrt_t1 = self.tet_points[v4] - t1
        vol = np.dot(t2_wrt_t1, np.cross(t3_wrt_t1, t4_wrt_t1))/6
        if not signed:
            return np.abs(vol)
        else:
            return vol

    #Returns the volume of the tet cluster
    def get_cluster_volume(self, cluster):
        total_volume = 0
        for t in cluster:
            total_volume += self.tet_volume(t)
        return total_volume

    #Returns the volume of the tet cluster
    def cluster_precomputed_volume(self, cluster_idx):
        #Compute the volume of a cluster only once.
        if cluster_idx in self.cluster_volume.keys():
            #If the cluster volume was already computed.
            return self.cluster_volume[cluster_idx]
        else:
            #If the cluster volume has not yet been computed, sum the
            #volume of its children and assign it to the cluster.
            total_volume = 0
            for c in self.children[cluster_idx]:
                vol = self.cluster_precomputed_volume(c)
                self.cluster_volume[c] = vol
                total_volume += vol
            #Update so we do this only once
            self.cluster_volume[cluster_idx] = total_volume
            return total_volume

    #Computes the convex hull using QuickHull and returns its volume.
    #The input is a list of tetrahedra that makes the cluster.
    def cluster_convex_hull_volume(self, list_cluster_tet):
        #Compute the convex hull using QuickHull
        cluster_point_cloud = np.ndarray((len(list_cluster_tet)*4,3))
        for i,t in enumerate(list_cluster_tet):
            for j,p in enumerate(t):
                cluster_point_cloud[i*4+j] = self.tet_points[p]
        hull = ConvexHull(cluster_point_cloud)
        return hull.volume

    #The cost of a cluster is a measure of its concavity
    def compute_cluster_cost(self, cluster_idx):
        cluster_list    = self.clusters[cluster_idx]
        tet             = [self.tet_list[cluster] for cluster in cluster_list]
        cvx_hull_volume = self.cluster_convex_hull_volume(self.tet_list)
        cluster_volume  = self.cluster_precomputed_volume(cluster_idx)
        cost = cvx_hull_volume - cluster_volume
        #The cost is inverted because SortedKeyList pop() retrieve the largest.
        if cost > 0:
            return -(cost+1)
        else:
            n1 = len(tet)
            return -(n1**2/self.nb_tet**2)

    #Implements the Möller and Trumbore intersection algorithm
    #to find if the line joining two vertices (v1 and v2) intersects the shared face
    #described by the triangle (tri_1, tri_2, tri_3) along with the location of the intersection.
    #https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    def triangle_intersect(self, v1, v2, tri_1, tri_2, tri_3):
        #Any point in the triangle can be located using a combination of two bases
        e1 = tri_2 - tri_1
        e2 = tri_3 - tri_1
        #Plane supporting the triangle
        n   = np.cross(e1, e2)
        #Projection of the line joining the two vertices onto the plane
        det = -np.dot(v2-v1, n)
        #Do some magic I dont fully understand
        x = np.cross(v1-tri_1, v2-v1)
        u =  np.dot(e2,x)/det
        v = -np.dot(e1,x)/det
        t =  np.dot(v1-tri_1,n)/det
        #Test if the line segment intersects the triangle
        intersects = (det >= 1e-6 and t >= 0 and u >= 0 and v >= 0 and (u+v) <= 1)
        #Location of the intersection
        intersection = v1 + t*(v2-v1)
        return intersects, intersection

    #Efficiently compute the concavity of a cluster made of two tet
    # by computing the volume of the tet that would be built by joining the two peaks
    def two_tet_cost(self, tet1, tet2):
        all_vertices = np.hstack((tet1,tet2)).tolist()
        peak_t1  = None
        peak_t2  = None
        for v in tet1:
            if all_vertices.count(v) == 1:
                peak_t1 = v
                break
        for v in tet2:
            if all_vertices.count(v) == 1:
                peak_t2 = v
                break
        #The shared face is made of the v_shared vertices
        v_shared = list(set(all_vertices) - set([peak_t1]) - set([peak_t2]))
        #Test if the line segment joining the two peaks intersects the shared triangular face
        convex, intersection = self.triangle_intersect(self.tet_points[peak_t1], 
                                                        self.tet_points[peak_t2], 
                                                        self.tet_points[v_shared[0]], 
                                                        self.tet_points[v_shared[1]], 
                                                        self.tet_points[v_shared[2]])
        if not convex:
            dist_to_tri_v = np.linalg.norm(intersection - np.array([self.tet_points[v_shared[0]], 
                                                                    self.tet_points[v_shared[1]], 
                                                                    self.tet_points[v_shared[2]]]), axis=1)
            v1, v2 = set(range(0,3)) - set([np.argmax(dist_to_tri_v)])
            #The tetrahedron that would fill the concave space is define by peak_t1, peak_t2, v1 and v2
            concavity = self.tet_volume_from_points(peak_t1, peak_t2, v_shared[v1], v_shared[v2])
            return -(concavity+1)
        if convex:
            return -2/self.nb_tet**2

    #The cost of an edge is the measure of the concavity of the union of the clusters the edge links.
    def compute_edge_cost(self, edge):
        idx_cluster_1 = list(edge)[0]
        idx_cluster_2 = list(edge)[1]
        #Each cluster contains a list of tet
        cluster_1 = self.clusters[idx_cluster_1]
        cluster_2 = self.clusters[idx_cluster_2]
        #Concatenate the clusters to obtain the union of their tet
        union_cluster = cluster_1 + cluster_2
        union_cluster_tets = [self.tet_list[cluster] for cluster in union_cluster]
            
        #Convex hull volume
        cvx_hull_volume = self.cluster_convex_hull_volume(union_cluster_tets)

        #Compute the volume of the cluster union
        cluster_1_vol = self.cluster_precomputed_volume(idx_cluster_1)
        cluster_2_vol = self.cluster_precomputed_volume(idx_cluster_2)
        
        #If the convex hull has a greater volume than the tet volume, its concave.
        #The cost is inverted because SortedKeyList pop() retrieve the largest.
        cost = max(0, cvx_hull_volume - (cluster_1_vol+cluster_2_vol))
        if cost > 0:
            return -(cost+1)
        else:
            #Otherwise, the cluster is convex and we prioritize smaller clusters.
            n1 = len(cluster_1)
            n2 = len(cluster_2)
            return -(n1**2+n2**2)/self.nb_tet**2

    def cost_key(self, edge):
        return self.costs[edge]

    def run(self, initial_clusters=[], initial_labels=[]):
        start_time = time()
        
        self.build_dual_graph(initial_clusters, initial_labels)
        if self.profiling:
            print("Dual graph built in {} seconds.".format(time()-start_time))
            start_time = time()
        
        self.iterative_clustering()
        if self.profiling:
            print("Merging done in {} seconds.".format(time()-start_time))
            start_time = time()
        
        self.tree_pruning()
        if self.profiling:
            print("Tree pruning done in {} seconds.".format(time()-start_time))
            start_time = time()
        
    def build_dual_graph(self, initial_clusters=[], initial_labels=[]):

        #If an adequate initial clustering is provided, use it.
        if len(initial_clusters) > 0 and len(initial_labels) == len(self.tet_list):
            self.clusters = initial_clusters
            #Find the faces that connects tet from two different clusters
            clusters_connections = set()
            for i,c in enumerate(self.clusters):
                #Each cluster c contains a list of tet indices t_idx
                for j,t_idx in enumerate(c):
                    t = self.tet_list[t_idx]
                    faces = [t[0:3], t[1:4], np.hstack((t[0],t[2:4])), np.hstack((t[0:2],t[3]))]
                    #Find all tet sharing a face with t
                    neighbors = set()
                    for f in faces:
                        t2,ind = self.tet_search.find_face_neighbor(f,t)
                        if type(t2) != type(None):
                            ind = set(ind) - set([t_idx])
                            neighbors.add(list(ind)[0])
                    #Find neighbors with a different label
                    for neighbor in neighbors:
                        if initial_labels[neighbor] != initial_labels[i]:
                            #Find the cluster in which neighbor is
                            for k,cc in enumerate(self.clusters):
                                if neighbor in cc:
                                    if i != k:
                                        clusters_connections.add(frozenset([i, k]))

            #Compute the volumes of the clusters
            for i,c in enumerate(self.clusters):
                v = self.get_cluster_volume(c)
                self.cluster_volume[i] = v
                #Each cluster is a children
                self.children.append([i])

            #Create edges & vertices
            self.vertex_edges = [[] for c in self.clusters]#clusters_connections
            for i,conn in enumerate(clusters_connections):
                cluster_1, cluster_2 = conn
                self.edges.append([cluster_1,cluster_2])
                cost = self.compute_edge_cost([cluster_1, cluster_2])
                self.costs.append(cost)
                new_edge_idx = len(self.edges)-1
                self.edge_indices.add(new_edge_idx)
                self.vertex_edges[cluster_1].append(new_edge_idx)
                self.vertex_edges[cluster_2].append(new_edge_idx)

        #If no initial clustering is provided, build the dual graph from scratch
        else:
            for i,t1 in enumerate(self.tet_list):
                #Initialize a singleton cluster
                self.children.append([i])
                self.clusters.append([i])
                #Compare the vertices with all the other tet
                faces = [t1[0:3], t1[1:4], np.hstack((t1[0],t1[2:4])), np.hstack((t1[0:2],t1[3]))]
                
                #Find all tet sharing a face with t1
                neighbors = set()
                for f in faces:
                    t2,ind = self.tet_search.find_face_neighbor(f,t1)
                    if type(t2) != type(None):
                        ind = set(ind) - set([i])
                        neighbors.add(list(ind)[0])
                
                #Initialize an edge with all neighbors
                for n in neighbors:
                    #Since we iterate over tet in ascending order, if the neighbor has
                    # an index lower than i, it means the edge was already added (its undirected)
                    if i < n:
                        self.edges.append([i,n])
                        cost = self.two_tet_cost(self.tet_list[i], self.tet_list[n])
                        self.costs.append(cost)
                        new_edge_idx = len(self.edges)-1
                        self.edge_indices.add(new_edge_idx)
                        self.vertex_edges[i].append(new_edge_idx)
                        self.vertex_edges[n].append(new_edge_idx)

                #Pre-compute the volume of the tet
                t1_vol = self.tet_volume(i)
                self.cluster_volume[i] = t1_vol

    def iterative_clustering(self):
        cvx_hull_count = 0
        while len(self.edge_indices) > 0:
            #Pop the lowest cost edge in the list of 'dual edges' and merge the dual nodes/clusters
            #SortedKeyList pop() operation returns the LARGEST, requiring us to invert the cost.
            idx_edge_to_merge = self.edge_indices.pop()
            clusters_connected_with_edge  = self.edges[idx_edge_to_merge]
            #Indices of the clusters to be merged
            idx_cluster_1       = clusters_connected_with_edge[0]
            idx_cluster_2       = clusters_connected_with_edge[1]
            #If the edge connects clusters that were previously merged (and therefore have a parent), the edge is invalid.
            if idx_cluster_1 in self.parent.keys() or idx_cluster_2 in self.parent.keys():
                continue
            #Create a new cluster that will contain cluster_1 and cluster_2
            new_cluster_tet = self.clusters[idx_cluster_1] + self.clusters[idx_cluster_2]
            new_cluster_idx = len(self.clusters)
            self.clusters.insert(new_cluster_idx, new_cluster_tet)
            #Update the binary tree with the parent relationship
            self.children.insert(new_cluster_idx, [idx_cluster_1, idx_cluster_2])
            self.parent[idx_cluster_1] = new_cluster_idx
            self.parent[idx_cluster_2] = new_cluster_idx
            #Edges connecting to clusters to be merged
            connected_edges = self.vertex_edges[idx_cluster_1] + self.vertex_edges[idx_cluster_2]
            #Keep only unique edges and dont include idx_edge_to_merge
            connected_edges = list(set(connected_edges) - set([idx_edge_to_merge]) - (set(self.vertex_edges[idx_cluster_1]) & set(self.vertex_edges[idx_cluster_2])))
            #Verify that the theoretical maximum of connected_edges is respected.
            nb_tet = len(new_cluster_tet)
            max_nb_edges = 4*nb_tet - 2*(nb_tet-1)
            assert(len(connected_edges) <= max_nb_edges)
            #When a new cluster is created, a new vertex is also created.
            # The vertice that were merged are removed.
            self.vertex_edges.insert(new_cluster_idx,connected_edges)
            self.vertex_edges[idx_cluster_1] = []
            self.vertex_edges[idx_cluster_2] = []
            # And the edge that is contracted is also removed
            self.edges[idx_edge_to_merge] = []
            #Update all edges that were connected to merged vertice
            for i in connected_edges:
                e = self.edges[i]
                #Get the vertex that needs to be updated
                vert_update = set(clusters_connected_with_edge) & set(e)
                #Do not proceed if the connected_edge is the edge_to_merge, in which case vert_update will contain 2 elements
                if len(vert_update) == 1:
                    #Remove the edge before changing its cost because SortedKeyList requires that
                    # "The total ordering of values must not change while they are stored in the sorted list."
                    assert(i in self.edge_indices)
                    self.edge_indices.remove(i)
                    #Update the edge by replacing vert_update with new_cluster_idx
                    self.edges[i] = list((set(e)-vert_update) | set([new_cluster_idx]))
                    #Update the cost of the edge
                    self.costs[i] = self.compute_edge_cost(self.edges[i])
                    #Add the updated edge/cost
                    self.edge_indices.add(i)
                    cvx_hull_count += 1
        if self.profiling:
            print("Perfomed {} convex hull operations.".format(cvx_hull_count))

    def tree_pruning(self):
        #Get the root
        root_idx = len(self.clusters)-1

        #Prune the tree
        roots = [root_idx]
        for p in roots:
            #Mark this parent as 'visited'
            roots.remove(p)
            new_children = []
            cost_parent = self.compute_cluster_cost(p)
            for c in self.children[p]:
                #If the children is not a leaf/tet but is a cluster,
                #consider removing it.
                if c >= len(self.tet_list):
                    #If the children has a bigger cost than the parent, remove it.
                    cost_children = self.compute_cluster_cost(c)
                    if cost_children > cost_parent:
                        #Link children of c to parent of c
                        for cc in self.children[c]:
                            self.parent[cc] = p
                            new_children.append(cc)
                    #Otherwise, keep it.
                    else:
                        new_children.append(c)
                else:
                    #Otherwise, keep it.
                    new_children.append(c)
            #Update the children, with the pruned ones ommitted
            self.children[p] = new_children
            #Next, visit the children that are not leafs/tet
            roots += [c for c in self.children[p] if c >= len(self.tet_list)]
        self.root_idx = root_idx

    def dendrogram_clusters(self, level=2):
        #Retrieve the N-th level clusters. 
        # Each cluster can be considered a "part" in the part segmentation.
        parent_clusters = [self.root_idx]
        for l in range(0, level):
            new_parent_clusters = []
            for p in parent_clusters:
                for c in self.children[p]:
                    new_parent_clusters.append(c)
            parent_clusters = new_parent_clusters
        return parent_clusters

    def visualize_dendrogram(self, max_level=4):
        #Initialize the graph
        graph = pydot.Dot("dendrogram", graph_type="digraph", bgcolor="white")
        graph.add_node(pydot.Node(str(self.root_idx), shape="circle"))
        
        #Since the tree is binary, we can predict how many nodes need to be explored
        #such that max_level is fully explored.
        max_nodes_to_explore = sum([2**i for i in range(0,max_level)])
        nb_explored_nodes    = 1

        #Breadth-first exploration
        branches_to_be_explored = [self.root_idx]
        while len(branches_to_be_explored) > 0 and nb_explored_nodes < max_nodes_to_explore:
            p = branches_to_be_explored[0]
            for c in self.children[p]:
                #If the children is not a leaf, it will need to be explored.
                if c >= len(self.tet_list):
                    branches_to_be_explored.append(c)
                #Add a children node
                graph.add_node(pydot.Node(str(c), shape="circle"))
                graph.add_edge(pydot.Edge(str(p), str(c), color="black"))
                
                nb_explored_nodes = nb_explored_nodes + 1
            branches_to_be_explored.remove(p)
        #Write the output to a .dot file
        graph.write_raw("dendrogram.dot")

    def get_clustered_centroids(self, clusters):
        '''
        Returns a list of clusters where each cluster contains a list of tet centroids
        in columns 0-3, the segementation id in column 4 and the tet volumes in column 5.
        '''
        centroids = []
        #For each cluster, sample the centroid of every associated tet
        for i,cluster_idx in enumerate(clusters):
            cluster_tet_list = self.clusters[cluster_idx]
            list_centroids   = np.ndarray((len(cluster_tet_list),5))
            #Go over every tet in the cluster
            for j,t in enumerate(cluster_tet_list):
                #Get the vertices making up the tet
                points_indice = self.tet_list[t]
                points = np.array([self.tet_points[point] for point in points_indice])
                #Centroid is the mean of the vertices
                centroid = np.mean(points, axis=0)
                tet_volume = self.tet_volume(t)
                list_centroids[j,0:3] = centroid    #Centroid
                list_centroids[j,3]   = i           #Segment ID
                list_centroids[j,4]   = tet_volume  #Volume
            centroids.append(list_centroids)
        return centroids
