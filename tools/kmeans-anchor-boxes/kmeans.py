import numpy as np
import pdb
import torch

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps
# def iou(box, clusters):
#     """
#     Calculates the Intersection over Union (IoU) between a box and k clusters.
#     :param box: tuple or array, shifted to the origin (i. e. width and height)
#     :param clusters: numpy array of shape (k, 4) where k is the number of clusters
#     :return: numpy array of shape (k, 0) where k is the number of clusters
#     """
#     x = np.minimum(clusters[:, 0], box[0])
#     y = np.minimum(clusters[:, 1], box[1])
#     if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
#         raise ValueError("Box has no area")

#     intersection = x * y
#     box_area = box[0] * box[1]
#     cluster_area = clusters[:, 0] * clusters[:, 1]

#     iou_ = intersection / (box_area + cluster_area - intersection)

#     return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    boxes = torch.from_numpy(boxes).float().cuda()
    return torch.mean(iou(boxes, clusters))

def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    boxes = torch.from_numpy(boxes).float().cuda()
    clusters = torch.from_numpy(clusters).float().cuda()
    last_clusters = torch.from_numpy(last_clusters).long().cuda()
    count = 0
    while True:
        distances = 1 - iou(boxes, clusters)

        _, nearest_clusters = torch.min(distances, 1)
        tmp = len(last_clusters) - (last_clusters == nearest_clusters).sum() 
        print(count, tmp)
        if tmp == 0:
            break

        for cluster in range(k):
            clusters[cluster] = torch.mean(boxes[nearest_clusters == cluster], dim=0)

        last_clusters = nearest_clusters
        count += 1
    return clusters
