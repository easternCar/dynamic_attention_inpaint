import os
import torch
import yaml
import random
import numpy as np
from PIL import Image
import copy
from skimage.measure import block_reduce
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import cv2

import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.nn import Parameter


#  [ l t h w ]
# [PART_TYPE]
# 1 (RIGHT EYE)
# 2 (LEFT EYE)
# 3 (EYES)
# 4 (LEFT FACE)
# 5 (RIGHT FACE)
# 6 (LOWER FACE)
FACIAL_PARTIAL_MASKS = [
    [60, 41, 28, 33], # RIGHT EYE
    [20, 41, 28, 33], # LEFT EYE
    [14, 41, 25, 80], # EYES
    [18, 39, 53, 33], # LEFT FACE
    [58, 41, 53, 33], # RIGHT FACE
    [15, 70, 37, 85] # LOWER FACE
]


def pil_loader(path, chan='RGB'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(chan)


def default_loader(path, chan='RGB'):
    return pil_loader(path, chan)


def tensor_img_to_npimg(tensor_img):
    """
    Turn a tensor image with shape CxHxW to a numpy array image with shape HxWxC
    :param tensor_img:
    :return: a numpy array image with shape HxWxC
    """
    if not (torch.is_tensor(tensor_img) and tensor_img.ndimension() == 3):
        raise NotImplementedError("Not supported tensor image. Only tensors with dimension CxHxW are supported.")
    npimg = np.transpose(tensor_img.numpy(), (1, 2, 0))
    npimg = npimg.squeeze()
    assert isinstance(npimg, np.ndarray) and (npimg.ndim in {2, 3})
    return npimg


# Change the values of tensor x from range [0, 1] to [-1, 1]
def normalize(x):
    return x.mul_(2).add_(-1)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks






def load_predefined_mask(config, batch_size):
    img_height, img_width, _ = config['image_shape']
    

    mask_files = os.listdir(config['generated_mask_path'])
    mask_size = config['image_shape'][1]
    MASK_NUM = len(mask_files)

    # sampled N
    chosen = random.sample(range(0, MASK_NUM), batch_size)

    masks = []
    
    for i in range(batch_size):
        BINARY_MASK = cv2.imread(config['generated_mask_path'] + '/' + mask_files[chosen[i]], cv2.IMREAD_GRAYSCALE)
        BINARY_MASK = cv2.resize(BINARY_MASK, (mask_size, mask_size), cv2.cv2.INTER_CUBIC)
        thresh, BINARY_MASK = cv2.threshold(BINARY_MASK, 128, 255, cv2.THRESH_BINARY)

        masks.append(BINARY_MASK)

    return masks



    

def random_bbox(config, batch_size):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including img

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width, _ = config['image_shape']
    h, w = config['mask_shape']
    margin_height, margin_width = config['margin']
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []

    if config['center_mask'] == True:
        t = h/2
        l = w/2
        bbox_list.append((t, l, h, w))
        bbox_list = bbox_list * batch_size

    else:
        if config['mask_batch_same']:
            t = np.random.randint(margin_height, maxt)
            l = np.random.randint(margin_width, maxl)
            bbox_list.append((t, l, h, w))
            bbox_list = bbox_list * batch_size
        else:
            for i in range(batch_size):
                t = np.random.randint(margin_height, maxt)
                l = np.random.randint(margin_width, maxl)
                bbox_list.append((t, l, h, w))


    return torch.tensor(bbox_list, dtype=torch.int64)


# MY implemented function, facial component mask
def random_partial_bbox(config, batch_size, PART_TYPE=0):

    img_height, img_width, _ = config['image_shape']
    #h, w = config['mask_shape']
    bbox_list = []



    # [PART_TYPE]
    # 0 (RANDOM)
    # 1 (RIGHT EYE) [20, 25]
    # 2 (LEFT EYE) [20, 25]
    # 3 (EYES) [20, 60]
    # 4 (LEFT FACE) [53, 25]
    # 5 (RIGHT FACE) [53, 25]
    # 6 (LOWER FACE) [32, 60]

    if PART_TYPE == 0:
        #random choose
        PART_TYPE = np.random.randint(1, 7)


    if PART_TYPE == 1:
        this_mask = copy.deepcopy(FACIAL_PARTIAL_MASKS[0])
    elif PART_TYPE == 2:
        this_mask = copy.deepcopy(FACIAL_PARTIAL_MASKS[1])
    elif PART_TYPE == 3:
        this_mask = copy.deepcopy(FACIAL_PARTIAL_MASKS[2])
    elif PART_TYPE == 4:
        this_mask = copy.deepcopy(FACIAL_PARTIAL_MASKS[3])
    elif PART_TYPE == 5:
        this_mask = copy.deepcopy(FACIAL_PARTIAL_MASKS[4])
    else:
        this_mask = copy.deepcopy(FACIAL_PARTIAL_MASKS[5])

    h, w = this_mask[2], this_mask[3]


    FOR_TEST = True

    # fixed : automatically the batch size same
    if config['mask_partial_fixed']:

        if FOR_TEST:
            for idx in range(batch_size):
                #margin_pos_t = np.random.randint(-15, 15)
                #margin_pos_l = np.random.randint(-15, 15)
                margin_pos_t = 0
                margin_pos_l = 0

                l = this_mask[0] + margin_pos_l
                t = this_mask[1] + margin_pos_t
                mask_center_x = l + int(this_mask[3] / 2)
                mask_center_y = t + int(this_mask[2] / 2)

                new_l, new_t = make_safe_left_top(l, t, h, w, mask_center_x, mask_center_y, img_width, img_height)

                bbox_list.append((new_t, new_l, h, w))

        else:
            # mask L T H W
            # added in 2020/07
            if PART_TYPE == 1:
                this_mask = this_mask
            elif PART_TYPE == 2:
                this_mask = this_mask
            elif PART_TYPE == 3:
                this_mask[2] = int(this_mask[2] * 1.3)
                h, w = this_mask[2], this_mask[3]
            elif PART_TYPE == 4:
                this_mask[0] = this_mask[0] - 10
                this_mask[3] = int(this_mask[3] * 1.5)
                h, w = this_mask[2], this_mask[3]
            elif PART_TYPE == 5:
                this_mask[0] = this_mask[0] - 10
                this_mask[3] = int(this_mask[3] * 1.5)
                h, w = this_mask[2], this_mask[3]


            l = this_mask[0]
            t = this_mask[1]
            mask_center_x = l + int(this_mask[3] / 2)
            mask_center_y = t + int(this_mask[2] / 2)

            new_l, new_t = make_safe_left_top(l, t, h, w, mask_center_x, mask_center_y, img_width, img_height)
            new_l = l
            new_t = t
            bbox_list.append((new_t, new_l, h, w))
            bbox_list = bbox_list * batch_size

    else:
        if config['mask_batch_same']:
            margin_pos_t = np.random.randint(-15, 15)
            margin_pos_l = np.random.randint(-15, 15)

            l = this_mask[0] + margin_pos_l
            t = this_mask[1] + margin_pos_t
            mask_center_x = l + int(this_mask[3] / 2)
            mask_center_y = t + int(this_mask[2] / 2)

            new_l, new_t = make_safe_left_top(l, t, h, w, mask_center_x, mask_center_y, img_width, img_height)
            bbox_list.append((new_t, new_l, h, w))
            bbox_list = bbox_list * batch_size


        else:
            # every images are have differenct margins
            for idx in range(batch_size):
                margin_pos_t = np.random.randint(-15, 15)
                margin_pos_l = np.random.randint(-15, 15)

                l = this_mask[0] + margin_pos_l
                t = this_mask[1] + margin_pos_t
                mask_center_x = l + int(this_mask[3] / 2)
                mask_center_y = t + int(this_mask[2] / 2)

                new_l, new_t = make_safe_left_top(l, t, h, w, mask_center_x, mask_center_y, img_width, img_height)

                bbox_list.append((new_t, new_l, h, w))



    #print(bbox_list)
    return torch.tensor(bbox_list, dtype=torch.int64)


# get mask center, bbox data and image size
def make_safe_left_top(l, t, h, w, mask_center_x, mask_center_y, img_w, img_h):
    # l and t for bbox (
    t = mask_center_y - int(h / 2)
    l = mask_center_x - int(w / 2)

    if t < 0:
        t = 0
    if l < 0:
        l = 0

    # check if bbox is bigger than image
    surplus_y = 0
    surplus_x = 0
    if (t + h) >= img_h:
        surplus_y = (t + h) - img_h
    if (l + w) >= img_w:
        surplus_x = (l + w) - img_w

    l = l - surplus_x
    t = t - surplus_y

    return l, t


def test_random_bbox():
    image_shape = [256, 256, 3]
    mask_shape = [128, 128]
    margin = [0, 0]
    bbox = random_bbox(image_shape)
    return bbox


def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w, min_delta_h=0, min_delta_w=0):
    # bboxes = [
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)

    for i in range(batch_size):
        bbox = bboxes[i]

        delta_h = np.random.randint(min_delta_h, max_delta_h // 2 + 1)
        delta_w = np.random.randint(min_delta_w, max_delta_w // 2 + 1)
        mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.

    return mask


def test_bbox2mask():
    image_shape = [256, 256, 3]
    mask_shape = [128, 128]
    margin = [0, 0]
    max_delta_shape = [32, 32]
    bbox = random_bbox(image_shape)
    mask = bbox2mask(bbox, image_shape[0], image_shape[1], max_delta_shape[0], max_delta_shape[1])
    return mask


def local_patch(x, bbox_list):
    assert len(x.size()) == 4
    patches = []

    for i, bbox in enumerate(bbox_list):
        t, l, h, w = bbox
        patches.append(x[i, :, t:t + h, l:l + w])


    return torch.stack(patches, dim=0)


# call when using predefined mask
def mask_image_predefined(x, masks, config):

    height, width, _ = config['image_shape']

    # masks : list of cv2 ndarray
    batch_size = len(masks)

    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)

    for i in range(batch_size):
        mask[i, :, :, :] = torch.Tensor(masks[i]) / 255.0

    if x.is_cuda:
        mask = mask.cuda()

    if config['mask_type'] == 'hole':
        result = x * (1. - mask)

    return result, mask


# this is independent to bbox! (bbox is larger than mask)
def mask_image(x, bboxes, config):
    # image width, height
    height, width, _ = config['image_shape']

    # delta shape : random [0 ~ delta/2]
    max_delta_h, max_delta_w = config['max_delta_shape']

    if config['center_mask'] == False:
        # partial mask : min cannot be zero
        min_delta_h = max_delta_h // 4
        min_delta_w = max_delta_w // 4

        # if length is not 4
        '''
        mask = torch.zeros((bboxes.size()[0], 1, height, width), dtype=torch.float32)
        mask_num = int(bboxes.size()[1] / 4)

        for mask_idx in range(mask_num):
            bbox = bboxes[:, mask_idx*4:(mask_idx+1)*4]
            mask = mask * bbox2mask(bbox, height, width, max_delta_h, max_delta_w, min_delta_h, min_delta_w)

        '''

        mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w, min_delta_h, min_delta_w)

    else:
        mask = bbox2mask(bboxes, height, width, 0, 0)

    if x.is_cuda:
        mask = mask.cuda()

    if config['mask_type'] == 'hole':
        result = x * (1. - mask)
        #result = mask * torch.rand_like(mask) + x * (1. - mask)

    elif config['mask_type'] == 'mosaic':
        # TODO: Matching the mosaic patch size and the mask size
        mosaic_unit_size = config['mosaic_unit_size']
        downsampled_image = F.interpolate(x, scale_factor=1. / mosaic_unit_size, mode='nearest')
        upsampled_image = F.interpolate(downsampled_image, size=(height, width), mode='nearest')
        result = upsampled_image * mask + x * (1. - mask)
    else:
        raise NotImplementedError('Not implemented mask type.')

    return result, mask


def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    # gamma : 0.9 (default)
    gamma = config['spatial_discounting_gamma']

    # mask_shape, requires height, width
    # if free, then we need to know x1, y1, x2, y2 in binary mask
    height, width = 128, 128

    # [1, 1, h, w]
    shape = [1, 1, height, width]


    if config['discounted_mask']:
        mask_values = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                mask_values[i, j] = max(
                    gamma ** min(i, height - i),
                    gamma ** min(j, width - j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 0)
    else:
        mask_values = np.ones(shape)
    spatial_discounting_mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
    if config['cuda']:
        spatial_discounting_mask_tensor = spatial_discounting_mask_tensor.cuda()
    return spatial_discounting_mask_tensor


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


# make corelation map using RGB colors
def make_corel_map(corel_maps, K=8, option='order'):
    out = []

    # corel_resize

    # print("max : " + str(np.max(corel_maps)) + ", min : " + str(np.min(corel_maps)))
    #corel_maps = block_reduce(corel_maps, (1, 2, 2), np.mean)

    PATCH_NUM = corel_maps.shape[1] * corel_maps.shape[2]
    VAL_RANGE_COLOR = np.arange(0.0, 255.0, 255.0 / PATCH_NUM)
    VAL_INTERVAL = 255.0 / PATCH_NUM
    VAL_FACTOR = 3.0
    COLOR_ARR = np.zeros([PATCH_NUM, 3])

    for i in range(PATCH_NUM):
        if i < int(PATCH_NUM / 3):
            COLOR_ARR[i, :] = [(VAL_INTERVAL * i)/VAL_FACTOR, (VAL_INTERVAL * i)/VAL_FACTOR, 255.0]

        else :
            COLOR_ARR[i, :] = [(VAL_INTERVAL * i) * 1.5, (VAL_INTERVAL * i) * 1.5, 255.0]
    np.clip(COLOR_ARR, 0, 255.0, out=COLOR_ARR)

    # most similar patch denote as deep brown
    K = 0
    for i in range(K):
        COLOR_ARR[i, :] = [255.0, (i * 5.0), (i * 5.0)]

    if option == 'order':
        for i in range(corel_maps.shape[0]):
            corel_map = corel_maps[i, :, :]
            #max_corel_score = np.max(corel_map.ravel())

            # sort (values and indices)
            #corel_values_sorted = np.sort(corel_map.ravel())
            corel_indices = np.argsort(corel_map.ravel())

            # exclude low similar patches
            corel_indices[corel_indices > int(PATCH_NUM/1.5)] = (PATCH_NUM-1)
            #corel_indices[corel_map[corel_indices] < (max_corel_score/10)] = (PATCH_NUM-1)

            #print(corel_indices.shape)
            #print(COLOR_ARR.shape)

            corel_map_order = COLOR_ARR[corel_indices]
            final_corel_map = np.reshape(corel_map_order, (corel_maps.shape[1], corel_maps.shape[2], 3))

            #print(final_corel_map[0:4, 0:4, :])
            out.append(final_corel_map)

        # each image
    elif option == 'intensity':
        for i in range(corel_maps.shape[0]):
            corel_map = corel_maps[i, :, :]

            corel_max = np.max(corel_map.ravel())
            corel_min = np.min(corel_map.ravel())



            corel_indices = np.zeros([PATCH_NUM*PATCH_NUM])







    return np.float32(out)



def feature_map_visualize(fea):
    out = []
    RANGE_NUM = 100

    fea = fea.numpy()

    PATCH_NUM = fea.shape[1] * fea.shape[2]

    CMAPS = np.zeros([RANGE_NUM, 3])

    for i in range(int(RANGE_NUM/2)):
        CMAPS[i, :] = [0.0, 0.0 + (i*255.0/RANGE_NUM), 255.0 - (i*255.0/RANGE_NUM)]
    for i in range(int(RANGE_NUM/2), RANGE_NUM):
        CMAPS[i, :] = [0.0 + (i*255.0/RANGE_NUM), 255.0 - (i*255.0/RANGE_NUM), 0.0]

    for i in range(fea.shape[0]):
        fea *= RANGE_NUM
        fea = int(fea)
        fea = CMAPS[fea, :]

        im = Image.fromarray(fea)
        # im.save('./output/' + filenames[i])
        im.save(output_dir + filenames[i])




def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def pt_flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = torch.tensor(-999)
    maxv = torch.tensor(-999)
    minu = torch.tensor(999)
    minv = torch.tensor(999)
    maxrad = torch.tensor(-1)
    if torch.cuda.is_available():
        maxu = maxu.cuda()
        maxv = maxv.cuda()
        minu = minu.cuda()
        minv = minv.cuda()
        maxrad = maxrad.cuda()
    for i in range(flow.shape[0]):
        u = flow[i, 0, :, :]
        v = flow[i, 1, :, :]
        idxunknow = (torch.abs(u) > 1e7) + (torch.abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = torch.max(maxu, torch.max(u))
        minu = torch.min(minu, torch.min(u))
        maxv = torch.max(maxv, torch.max(v))
        minv = torch.min(minv, torch.min(v))
        rad = torch.sqrt((u ** 2 + v ** 2).float()).to(torch.int64)
        maxrad = torch.max(maxrad, torch.max(rad))
        u = u / (maxrad + torch.finfo(torch.float32).eps)
        v = v / (maxrad + torch.finfo(torch.float32).eps)
        # TODO: change the following to pytorch
        img = pt_compute_color(u, v)
        out.append(img)

    return torch.stack(out, dim=0)


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def pt_highlight_flow(flow):
    """Convert flow into middlebury color code image.
        """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def pt_compute_color(u, v):
    h, w = u.shape
    img = torch.zeros([3, h, w])
    if torch.cuda.is_available():
        img = img.cuda()
    nanIdx = (torch.isnan(u) + torch.isnan(v)) != 0
    u[nanIdx] = 0.
    v[nanIdx] = 0.
    # colorwheel = COLORWHEEL
    colorwheel = pt_make_color_wheel()
    if torch.cuda.is_available():
        colorwheel = colorwheel.cuda()
    ncols = colorwheel.size()[0]
    rad = torch.sqrt((u ** 2 + v ** 2).to(torch.float32))
    a = torch.atan2(-v.to(torch.float32), -u.to(torch.float32)) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = torch.floor(fk).to(torch.int64)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0.to(torch.float32)
    for i in range(colorwheel.size()[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1]
        col1 = tmp[k1 - 1]
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1. / 255.
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = (idx != 0)
        col[notidx] *= 0.75
        img[i, :, :] = col * (1 - nanIdx).to(torch.float32)
    return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def pt_make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 1.
    colorwheel[0:RY, 1] = torch.arange(0, RY, dtype=torch.float32) / RY
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 1. - (torch.arange(0, YG, dtype=torch.float32) / YG)
    colorwheel[col:col + YG, 1] = 1.
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 1.
    colorwheel[col:col + GC, 2] = torch.arange(0, GC, dtype=torch.float32) / GC
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 1. - (torch.arange(0, CB, dtype=torch.float32) / CB)
    colorwheel[col:col + CB, 2] = 1.
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 1.
    colorwheel[col:col + BM, 0] = torch.arange(0, BM, dtype=torch.float32) / BM
    col += BM
    # MR
    colorwheel[col:col + MR, 2] = 1. - (torch.arange(0, MR, dtype=torch.float32) / MR)
    colorwheel[col:col + MR, 0] = 1.
    return colorwheel


def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def deprocess(img):
    img = img.add_(1).div_(2)
    return img


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


# Get model list for resume
def get_model_list(dirname, key, iteration=0):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    if iteration == 0:
        last_model_name = gen_models[-1]
    else:
        for model_name in gen_models:
            if '{:0>8d}'.format(iteration) in model_name:
                return model_name
        raise ValueError('Not found models with this iteration')
    return last_model_name




# I defined this for saving each images with original filenames
def save_each_image_everything(tensor, output_dir, filenames, normalize=False):
    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

            norm_range(tensor, range)

    #for i in range(tensor.size(0)):
    for i in range(6):
        img = tensor[i]
        #ndarr = img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        ndarr = img.add_(1).mul_(127.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        #im.save('./output/' + filenames[i])
        im.save(output_dir + filenames[i])


# I defined this for saving each images with original filenames
def save_each_image(tensor, filenames, output_dir, normalize=False):

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

            norm_range(tensor, range)

    for i in range(tensor.size(0)):
        img = tensor[i]
        #ndarr = img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        ndarr = img.add_(1).mul_(127.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)


        # if filenames have '/'....
        filename = filenames[i]
        if "/" in filenames[i]:
            filename = filenames[i].split("/")[1]

        #im.save('./output/' + filenames[i])
        #im.save('./output_test_poster/' + filenames[i])
        im.save(output_dir + filename)




# I defined this for saving each images with original filenames
def save_each_image_map(tensor, filename):

    f, axarr = plt.subplots(tensor.size(0))

    for ax, x_ in zip(axarr, tensor.squeeze(1)):
        ax.imshow(x_.to('cpu', torch.float).numpy(), cmap='jet')


    f.set_size_inches(15.5, 30.5)
    f.savefig(filename)


def get_psnr(original, compared):
    #mse = np.mean(np.square(original - compared))
    #psnr = np.clip(np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]
    psnr = compare_psnr(original, compared)
    return psnr

def get_ssim(original, compared):
    (score, diff) = compare_ssim(original, compared)
    return score


def save_image_with_overlay(tensor, mask, viz_max_out, filename, nrow=8, padding=0,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    import torchvision.utils as vutils

    #print("overlay saving")
    #print(tensor.size())

    tensor_norm = tensor.clone()
    tensor_norm = tensor_norm.mul_(255).add_(0.5)

    input_img = tensor_norm[0:viz_max_out]
    inpainted_img = tensor_norm[viz_max_out:2*viz_max_out]
    corel_map = tensor_norm[2 * viz_max_out: 3 * viz_max_out]


    mask_rgb = (1.0-mask).mul_(255).repeat(1, 3, 1, 1)
    #mask_rgb = torch.cat([mask_rgb]*3)

    #print(torch.max(mask_rgb))


    #overlayed = input_img * 0.75 + corel_map * 0.25
    overlayed = mask_rgb * 0.45 + corel_map * 0.55
    overlayed = overlayed.clamp_(0,255).div_(255).sub_(0.5)

    #tensor = torch.cat([tensor, overlayed], dim=0)
    tensor = torch.cat([tensor, (1.0-mask).repeat(1,3,1,1), overlayed], dim=0)

    grid = vutils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer

    im = Image.fromarray(ndarr)
    im.save(filename)

def load_my_state_dict(our_model, state_dict):
    own_state = our_model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        own_state[name].copy_(param)



# hooking
class LayerResult:
    def __init__(self, payers, layer_index):

        if layer_index == -1:
            self.hook = payers.register_forward_hook(self.hook_fn)
        else:
            self.hook = payers[layer_index].register_forward_hook(self.hook_fn)
        self.outputs = []
    def hook_fn(self, module, input, output):
        #self.features = torch.stack(output, dim=1).cpu().data.numpy()
        #self.features = output.cpu().data.numpy()

        output = F.sigmoid(output)

        #if output.size()[2] != self.FIX_MAP_SIZE:
        #    output = torch.nn.functional.interpolate(output, size=[self.FIX_MAP_SIZE, self.FIX_MAP_SIZE], mode='bilinear')


        #self.features = output.cpu().data.numpy()
        self.features = output
    def unregister_forward_hook(self):
        self.hook.remove()

    def hook(self, module, input, output):
        self.outputs.append(output)



def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss





if __name__ == '__main__':
    test_random_bbox()
    mask = test_bbox2mask()
    print(mask.shape)
    import matplotlib.pyplot as plt

    plt.imshow(mask, cmap='gray')
    plt.show()
