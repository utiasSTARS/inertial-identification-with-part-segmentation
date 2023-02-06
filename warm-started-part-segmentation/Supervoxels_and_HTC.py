#The usage of supervoxels pre-processing can greatly lower the processing time
#by allowing an initial coarse clustering that limits the number of merge operations.
import numpy as np
import open3d as o3d
from time import time
from pathlib import Path
import matplotlib.pyplot as plt 
from HierarchicalTetClustering import HierarchicalTetClustering
from SupervoxelSegmentation import Supervoxels

#Compute point normals
def compute_point_normals(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=10))
    pcd.orient_normals_consistent_tangent_plane(10)
    return pcd

#Scale & Center
def scale_and_center(pcd):
    bb_extent = pcd.get_oriented_bounding_box().extent
    bb_center = pcd.get_oriented_bounding_box().center
    bb_diag   = np.sqrt(bb_extent[0]**2 + bb_extent[1]**2 + bb_extent[2]**2)
    pcd.translate(-1*bb_center)
    pcd.scale(1/bb_diag, center=[0,0,0])
    print("Point cloud loaded. So far {} seconds passed.".format(time()-start_time))
    return bb_diag, bb_center

#Instanciate thread used to produce supervoxels
def get_supervoxels(pcd):
    supervoxels_thread = Supervoxels(pcd, 50, 10)
    supervoxels_thread.start()
    #Wait until result is ready
    supervoxels, point_label = supervoxels_thread.join()
    print("Supervoxels obtained. So far {} seconds passed.".format(time()-start_time))
    return supervoxels, point_label

#Use supervoxels to initialize HTC clusters
def initialize_htc(htc, supervoxels, point_label):
    #For each tetrahedron, find the nearest point from the point cloud
    #The label of the tetrahedron is the label of the point
    label_tet = []
    kd_tree   = o3d.geometry.KDTreeFlann(pcd)
    for i,t in enumerate(htc.tet_list):
        centroid = np.mean([htc.tet_points[p] for p in t], axis=0)
        [k, idx, _] = kd_tree.search_knn_vector_3d(centroid, 1)
        #Index in pcd of the nearest point to centroid of tet
        nearest_point_idx = idx[0]
        #Supervoxel label associated with the nearest point
        label_tet.insert(i, point_label[nearest_point_idx])
    
    #Initialize the tet clusters from the supervoxels
    clusters = [[] for s in supervoxels]
    for i,t in enumerate(htc.tet_list):
        j = label_tet[i]
        clusters[j].append(i)

    return clusters, label_tet

def run_htc(htc, clusters, label_tet, level):
    '''
    Run the Tet Clustering from the initial state.
    Return a list of lists of tet centroids.
    '''
    htc.run(initial_clusters=clusters, initial_labels=label_tet)
    clusters = htc.dendrogram_clusters(level)
    clustered_centroids = htc.get_clustered_centroids(clusters)
    return clustered_centroids

def get_geometries(bb_diag, bb_center, clustered_centroids):
    #Visualize the segmentation
    part_point_cloud   = []
    object_points = np.ndarray((0,3))
    color_label   = plt.get_cmap("inferno")(np.linspace(0, 1, len(clustered_centroids)+1))[:,0:3]
    for i,cluster in enumerate(clustered_centroids):
        #Parse cluster
        points = cluster[:,0:3]
        #Build a point-cloud from all the centroids
        o3d_pcd         = o3d.geometry.PointCloud()
        o3d_pcd.points  = o3d.utility.Vector3dVector(points)
        #IMPORTANT: Unscale the result so the distances are consistent with the original point-cloud
        o3d_pcd.scale(bb_diag, center=[0,0,0])
        o3d_pcd.translate(bb_center)
        #Colorize
        point_label_color   = [color_label[i] for j in range(0, len(points))]
        o3d_pcd.colors      = o3d.utility.Vector3dVector(point_label_color)
        #Append the cluster point-cloud to the list
        part_point_cloud.append(o3d_pcd)
        object_points = np.vstack((object_points,o3d_pcd.points))
    
    return object_points, part_point_cloud

def save_segmentation(output_path:Path, part_point_cloud, clustered_centroids):
    '''
    Save a .PLY file where each point represent the centroid of a tetrahedra
    and colors represent segments.
    '''
    object_points = np.ndarray((0,3))
    object_colors = np.ndarray((0,3))
    object_segment= np.ndarray((0,1))
    object_volumes= np.ndarray((0,1))
    #Combine point clouds together
    for i,pcd in enumerate(part_point_cloud):
        #A cluster is a list of tet [centroid, segmentation id, volume]
        cluster = clustered_centroids[i]
        object_points = np.vstack((object_points, pcd.points))
        object_colors = np.vstack((object_colors, pcd.colors))
        object_segment= np.vstack((object_segment, np.tile(i,(len(pcd.points),1))))
        object_volumes= np.vstack((object_volumes, np.array([c[4]for c in cluster]).reshape((cluster.shape[0],1))))
    
    #Create an Open3D PointCloud object
    device = o3d.core.Device("CPU:0")
    o3d_pointcloud = o3d.t.geometry.PointCloud(device)
    o3d_pointcloud.point['points']       = o3d.core.Tensor(np.array(object_points), o3d.core.Dtype.Float32, device)
    o3d_pointcloud.point['colors']       = o3d.core.Tensor(np.array(object_colors), o3d.core.Dtype.Float32, device)
    o3d_pointcloud.point['segmentation'] = o3d.core.Tensor(np.array(object_segment), o3d.core.Dtype.UInt8, device)
    o3d_pointcloud.point['volume']       = o3d.core.Tensor(np.array(object_volumes), o3d.core.Dtype.Float32, device)

    #Write PointCloud
    output_path = Path(output_path)
    output_posix_path = "{}".format(output_path.resolve().as_posix())
    print('Writing {} points to {}'.format(len(o3d_pointcloud.point['points']), output_posix_path))
    o3d.t.io.write_point_cloud(output_posix_path, o3d_pointcloud, write_ascii=True)


TETGEN_PATH = Path('./tetgen1.6.0/build/tetgen')

OUTPUT_FOLDER     = Path("workshop_tools")
DATASET_BASE_PATH = Path("../data/Workshop Tools Dataset")
ply_files = DATASET_BASE_PATH.glob('*/point-cloud.ply')

for ply in ply_files:
    if ply.exists() and not ply.is_dir():
        ply_path = ply.as_posix()
        object_name = ply.parent.stem.replace(' ','_').replace('+','_')

        print('Segmenting {}'.format(object_name))

        start_time = time()

        pcd = o3d.io.read_point_cloud(ply_path)
        bb_diag, bb_center = scale_and_center(pcd)
        pcd = compute_point_normals(pcd)

        supervoxels, point_label = get_supervoxels(pcd)

        #Instanciate Hierarchical Tetrahedra Clustering
        htc = HierarchicalTetClustering(np.asarray(pcd.points), np.asarray(pcd.normals), TETGEN_PATH, profiling=True)
        clusters, label_tet = initialize_htc(htc, supervoxels, point_label)
        clustered_centroids = run_htc(htc, clusters, label_tet, level=2)

        print("Total time is {} seconds.".format(time()-start_time))

        object_points, part_point_cloud = get_geometries(bb_diag, bb_center, clustered_centroids)

        path = Path(OUTPUT_FOLDER/"output_segmentations/{}".format(object_name))
        path.mkdir(parents=True, exist_ok=True)
        save_segmentation(path/"{}.ply".format(object_name), part_point_cloud, clustered_centroids)
