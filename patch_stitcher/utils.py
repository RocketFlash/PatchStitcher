import cv2
from pytorch_toolbelt.inference.tiles import ImageSlicer
from matplotlib import pyplot as plt
import math
import torch
import numpy as np
from skimage import io
import kornia as K
import matplotlib 
import matplotlib.cm as cm


def crop(image, only_lr=False, only_ud=False):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    if only_lr:
        return image[:, np.min(x_nonzero):np.max(x_nonzero)]
    elif only_ud:
        return image[np.min(y_nonzero):np.max(y_nonzero), :]
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    
    
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])    
    return frame


def make_matching_figure(
        img0, img1, mkpts0, mkpts1, conf,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair

    color = cm.jet(conf)
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(20, 20), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig



def load_torch_image(image):
    return K.image_to_tensor(image, False).float() / 255.

def get_rotated_patch(image, coords, degrees):
    points, r_w, r_h = rotate_coords(coords, degrees)
    src_pts = points.astype("float32")
    dst_pts = np.array([[0, 0],
                        [r_w-1, 0],
                        [r_w-1, r_h-1],
                        [0, r_h-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (r_w, r_h))


def rotate_coords(coords, degrees=10):
    (x1, y1), (x2, y2) = coords
    r_w, r_h = x2-x1, y2-y1
    center = (int(x1+r_w/2), int(y1+r_h/2))
    pt1_ = [x1, y1]
    pt2_ = [x2, y1]
    pt3_ = [x2, y2]
    pt4_ = [x1, y2]

    radians = np.deg2rad(degrees)
    pt1 = rotate_around_point(pt1_, radians, origin=center)
    pt2 = rotate_around_point(pt2_, radians, origin=center)
    pt3 = rotate_around_point(pt3_, radians, origin=center)
    pt4 = rotate_around_point(pt4_, radians, origin=center)

    return np.array([pt1, pt2, pt3, pt4]), r_w, r_h


def rotate_around_point(point, radians, origin=(0, 0)):
    x, y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return int(qx), int(qy)


# def find_outliers(a, m=1.5):
#     upper_quartile = np.percentile(a, 75)
#     lower_quartile = np.percentile(a, 25)
#     IQR = (upper_quartile - lower_quartile) * m
#     quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    
#     result = (a >= quartileSet[0]) & (a <= quartileSet[1])
#     return result

# def find_outliers(a, m=2):
#     d = np.abs(a - np.median(a))
#     mdev = np.median(d)
#     s = d/mdev if mdev else 0.
#     return s<m

# def find_outliers(a, m=2):
#     return abs(a - np.median(a)) < m * np.std(a)

def find_outliers(a, m=2):
    return abs(a - np.median(a))<m


def get_device(device_str):
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device 

def xywh2xyxy(coords):
    x, y, w, h = coords
    return [(x,y), (x+w,y+h)]


def plot_tiles_similarity(similarity_matrix, save_path='./img.png'):
    fig, ax = plt.subplots( nrows=1, ncols=1 )

    ax.imshow(similarity_matrix, interpolation='none')
    ax.set_xlabel('source tile index')
    ax.set_ylabel('target tile index')

    fig.savefig(save_path)
    plt.close(fig)


def get_image(image_path):
    image = cv2.imread(str(image_path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def split_image_on_patches(image, window_size=20, step_size=10, is_horizontal=True, is_multidirect=False):
    h, w, c = image.shape

    if is_multidirect:
        tile_scale = 1/7
        tile_size = (window_size, int(window_size*tile_scale)) if is_horizontal else (int(window_size*tile_scale), window_size)
        tile_step = (int(window_size*tile_scale), step_size) if is_horizontal else (step_size, int(window_size*tile_scale)) 
    else:
        tile_size = (h, window_size) if is_horizontal else (window_size, w)
        tile_step = (1, step_size) if is_horizontal else (step_size, 1)
    tiler = ImageSlicer(image.shape, tile_size=tile_size, tile_step=tile_step)
    tiles = tiler.split(image)
    return tiles, tiler

def show_images(images, n_col=3, save_name=None):
    n_rows = math.ceil(len(images)/n_col)
    fig, ax = plt.subplots(n_rows, n_col, figsize=(25, 12*n_rows))

    for ax_i in ax:
        if len(images) <= n_col:
            ax_i.set_axis_off()
        else:
            for ax_j in ax_i:
                ax_j.set_axis_off()

    if isinstance(images, dict):
        for img_idx, (title, img) in enumerate(images.items()):
            if len(images) <= n_col:
                ax[img_idx].imshow(img)
                ax[img_idx].set_title(title)
            else:
                ax[img_idx//n_col, img_idx%n_col].imshow(img)
                ax[img_idx//n_col, img_idx%n_col].set_title(title)
    else:
        for img_idx, img in enumerate(images):
            if len(images) <= n_col:
                ax[img_idx].imshow(img)
            else:
                ax[img_idx//n_col, img_idx%n_col].imshow(img)

    fig.subplots_adjust(wspace=0, hspace=0)
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)
        plt.close(fig)