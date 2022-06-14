import cv2
import numpy as np


def crop_(panorama, h_dst, conners):
    [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
    t = [-xmin, -ymin]
    conners = conners.astype(int)

    if conners[0][0][0] < 0:
        n = abs(-conners[1][0][0] + conners[0][0][0])
        panorama = panorama[t[1] : h_dst + t[1], n:, :]
    else:
        if conners[2][0][0] < conners[3][0][0]:
            panorama = panorama[t[1] : h_dst + t[1], 0 : conners[2][0][0], :]
        else:
            panorama = panorama[t[1] : h_dst + t[1], 0 : conners[3][0][0], :]
    return panorama


def blending_mask(height, width, barrier, smoothing_window, left_biased=True):
    assert barrier < width
    mask = np.zeros((height, width))

    offset = int(smoothing_window / 2)
    try:
        if left_biased:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset + 1).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset + 1).T, (height, 1)
            )
            mask[:, barrier + offset :] = 1
    except BaseException:
        if left_biased:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset : barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height, 1)
            )
            mask[:, barrier + offset :] = 1

    return cv2.merge([mask, mask, mask])


def panorama_blending(dst_img_rz, src_img_warped, width_dst, side):
    h, w, _ = dst_img_rz.shape
    smoothing_window = int(width_dst / 8)
    barrier = width_dst - int(smoothing_window / 2)
    mask1 = blending_mask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=True
    )
    mask2 = blending_mask(
        h, w, barrier, smoothing_window=smoothing_window, left_biased=False
    )

    if side == "left":
        dst_img_rz = cv2.flip(dst_img_rz, 1)
        src_img_warped = cv2.flip(src_img_warped, 1)
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz
        pano = cv2.flip(pano, 1)
        
    else:
        dst_img_rz = dst_img_rz * mask1
        src_img_warped = src_img_warped * mask2
        pano = src_img_warped + dst_img_rz
        

    return pano


def stitch_two_images(s_img, t_img, M):
    h_t, w_t = t_img.shape[:2]
    h_s, w_s = s_img.shape[:2] 

    
    t_img_warped, pts, t = warp_target(s_img, t_img, M)
    height_pano, width_pano = t_img_warped.shape[:2]
    
    dst_img_rz = np.zeros((height_pano, width_pano, 3))
    dst_img_rz[t[1] : h_s + t[1], t[0] : w_s + t[0]] = s_img
    

    stitched_image  = panorama_blending(dst_img_rz, t_img_warped, w_s, side='left')
    stitched_image = crop_(stitched_image, h_s, pts).astype("uint8")

    return stitched_image


def warp_target(s_img, t_img, M, is_horizontal=True):
    h_t, w_t = t_img.shape[:2]
    h_s, w_s = s_img.shape[:2] 
    pts1 = np.float32([[0, 0], [0, h_t], [w_t, h_t], [w_t, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h_s], [w_s, h_s], [w_s, 0]]).reshape(-1, 1, 2)

    pts1_ = cv2.perspectiveTransform(pts1, M)
    pts = np.concatenate((pts1_, pts2), axis=0)

    [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
    
    t = [-xmin, -ymin]
    
    if is_horizontal:
        width_pano = w_s + t[0]
        height_pano = ymax - ymin
    else:
        width_pano = xmax - xmin
        height_pano = h_s + t[1]

    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    t_img_warped = cv2.warpPerspective(t_img, Ht.dot(M), (width_pano, height_pano))
    return t_img_warped, pts, t