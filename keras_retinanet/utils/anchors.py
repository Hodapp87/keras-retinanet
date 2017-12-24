"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np


def anchor_targets_bbox(
    image_shape,
    boxes,
    num_classes,
    mask_shape=None,
    negative_overlap=0.4,
    positive_overlap=0.5,
    **kwargs
):
    """Computes label encoding and bounding box regression targets, given
    some set of anchors for the feature pyramid of an input image of
    some shape.  That is: it computes overlap by Intersection over
    Union for each anchor compared to each ground truth object box,
    and is IoU > positive_overlap, this anchor is assigned to that
    bounding box and its respective object class; if IoU <
    negative_overlap, it is assigned to the background.  If IoU is in
    between negative_overlap and positive_overlap, the anchor is
    simply ignored.  This is the scheme described in the "Focal Loss
    for Dense Object Detection" paper (arXiv:1708.02002).

    Returns:
    (labels, bbox_reg_targets).
    'labels' is a NumPy array of shape (N, num_classes) for which
    labels[n,c] is 0 if anchor 'n' is *not* object class 'c', is 1 if
    it is object class 'c', or is -1 if it doesn't matter (e.g. it's
    outside of the image or doesn't overlap enough).
    'bbox_reg_targets' is a NumPy array of shape (N, 4) where each
    anchor corresponds to a row, and contains the bounding box
    regression targets in the form [tx,ty,tw,th] - see bbox_transform.

    Parameters:
    image_shape -- Input image dimensions as (height,width)
    boxes -- Annotations (ground truth object boxes) as [x1,y1,x2,y2,label]
    num_classes -- Number of object classes
    mask_shape -- (y,x) of lower-right corner of mask (top-left is (0,0))
    negative_overlap -- IoU threshold to mark as background (default 0.4)
    positive_overlap -- IoU threshold for positive example (default 0.5)
    **kwargs -- Pyramid parameters passed on to anchors_for_shape

    """
    anchors = anchors_for_shape(image_shape, **kwargs)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.ones((anchors.shape[0], num_classes)) * -1

    if boxes.shape[0]:
        # obtain indices of gt boxes with the greatest overlap
        overlaps             = compute_overlap(anchors, boxes[:, :4])
        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps         = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < negative_overlap, :] = 0

        # compute box regression targets
        boxes            = boxes[argmax_overlaps_inds]
        bbox_reg_targets = bbox_transform(anchors, boxes)

        # fg label: above threshold IOU
        positive_indices = max_overlaps >= positive_overlap
        labels[positive_indices, :] = 0
        labels[positive_indices, boxes[positive_indices, 4].astype(int)] = 1
    else:
        # no annotations? then everything is background
        labels[:] = 0
        bbox_reg_targets = np.zeros_like(anchors)

    # ignore boxes outside of image
    mask_shape         = image_shape if mask_shape is None else mask_shape
    anchors_centers    = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
    indices            = np.logical_or(anchors_centers[:, 0] >= mask_shape[1], anchors_centers[:, 1] >= mask_shape[0])
    labels[indices, :] = -1

    return labels, bbox_reg_targets


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None
):
    """Computes bounding boxes of the anchors at each pyramid level for a
    given input image shape and pyramid parameters.  Returns a NumPy
    array with one row per anchor, and each row equaling [x1,y1,x2,y2]
    where (x1,y1) is the top-left corner and (x2,y2) the bottom-right
    corner in the coordinate space of the input image.

    Parameters:
    image_shape -- (height, width) of input image
    pyramid_levels -- List of pyramid level used for feature maps;
                      default [3, 4, 5, 6, 7]
    ratios -- Which aspect ratios to make anchors for; default [0.5, 1, 2]
    scales -- Which scale factors to make anchors for; default
              [1, 2^(1/3), 2^(2/3)]
    strides -- A list giving, for each pyramid level, the distance in the
               input image for each motion in the feature map at that level.
               Should be same length as pyramid_levels.  Default is based on
               pyramid_levels.
    sizes -- A list giving, for each pyramid level, the base sidelength
             (i.e. at scale 1) of the anchor at that level.  Default is based
             on pyramid_levels.

    """
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]
    if strides is None:
        strides = [2 ** x for x in pyramid_levels]
    if sizes is None:
        sizes = [2 ** (x + 2) for x in pyramid_levels]
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    # skip the first two levels
    image_shape = np.array(image_shape[:2])
    for i in range(pyramid_levels[0] - 1):
        image_shape = (image_shape + 1) // 2

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        image_shape     = (image_shape + 1) // 2
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shape, strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """Generates relative coordinates of anchors by enumerating aspect
    ratios and scales from some base size.  This returns a NumPy array
    of shape (len(ratios)*len(scales), 4).

    Each row contains [x1,y1,x2,y2], giving coordinates of top-left
    and bottom-right corners of the anchor, relative to some reference
    window.

    Parameters:
    base_size -- Base sidelength for anchors (default 16)
    ratios -- List/array of aspect ratios (default [0.5, 1, 2])
    scales -- List/array of scale factors (default [1, 2^(1/3), 2^(2/3)])

    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Computes bounding-box regression targets.  Returns a NumPy array
    with the same shape as 'anchors' or 'gt_boxes', but with each row
    giving [tx, ty, tw, th].

    This parametrizes bounding boxes according to appendix C of the
    Fast R-CNN paper (arXiv 1311.2524v5), specifically, equations 6-9
    - or, equivalently, equation 2 of the Faster R-CNN paper (arXiv
    1506.01497).

    Parameters:
    anchors -- Array giving anchor coordinates as rows of [x0,y0,x1,y1]
    gt_boxes -- Array giving ground truth object boxes in same format.
                Should be same shape as 'anchors'.
    mean -- Optional 4-element array with respective means for [tx, ty, tw, th]
    std -- Optional 4-element array with standard deviations (same format)

    """

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.1, 0.1, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths  = anchors[:, 2] - anchors[:, 0] + 1.0
    anchor_heights = anchors[:, 3] - anchors[:, 1] + 1.0
    # Anchor center points:
    anchor_ctr_x   = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y   = anchors[:, 1] + 0.5 * anchor_heights

    gt_widths  = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    # Ground-truth object box center points:
    gt_ctr_x   = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y   = gt_boxes[:, 1] + 0.5 * gt_heights

    # Same as t_x, t_y, t_w, t_h in paper:
    targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
    targets_dw = np.log(gt_widths / anchor_widths)
    targets_dh = np.log(gt_heights / anchor_heights)

    # Produce rows of [tx, ty, tw, th]
    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh))
    targets = targets.T

    # TODO: Why is this used?
    targets = (targets - mean) / std

    return targets


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0]) + 1
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1]) + 1

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua
