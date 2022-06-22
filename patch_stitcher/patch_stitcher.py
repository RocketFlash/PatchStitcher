import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import xywh2xyxy, plot_tiles_similarity, find_outliers
from .data import TilesDataset
from .utils import get_image, show_images, rotate_coords, get_rotated_patch

import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import ndimage
import kornia as K
import kornia.feature as KF
from .utils import load_torch_image, make_matching_figure, crop, draw_visualization
from .utils import find_outliers_angle, find_outliers_distances, make_border
from .classical_stitching import stitch_two_images, warp_target
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from numpy.linalg import inv



class PatchStitcher:

    MIN_MATCH_COUNT = 15

    def __init__(self, model_path,
                       device='cpu',
                       output_images_scale=0.5,
                       window_size=500,
                       step_size=20,
                       batch_size=50,
                       num_workers=8,
                       vis_save_path='./',
                       use_loftr=False,
                       loftr_conf_thresh=0.5,
                       use_dhe=False,
                       use_metric=True,
                       use_sift=False,
                       use_kp_filtering=False,
                       save_vis=True,
                       pyramid_search=False,
                       fast_search=True):
        self.device = device
        
        if model_path is None:
            from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
            from torchvision import transforms

            self.transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(400, 200),
                                        np.float32,
                                        transforms.ToTensor(),
                                        fixed_image_standardization])
            self.model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        else:
            self.model = torch.jit.load(model_path, map_location=device).to(device).eval()
            self.transform = A.Compose([A.Resize(400, 200),
                                        A.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), p=1),
                                        A.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                        ToTensorV2()])

    
        self.window_size = window_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.num_workers = num_workers
        self.o_s = output_images_scale
        self.vis_save_path = Path(vis_save_path)
        self.save_steps_vis = save_vis
        self.use_loftr = use_loftr
        self.loftr_conf_thresh = loftr_conf_thresh
        self.use_kp_filtering = use_kp_filtering
        self.use_dhe = use_dhe
        self.use_metric = use_metric
        self.use_sift = use_sift
        self.pyramid_search = pyramid_search
        self.fast_search = fast_search

        if self.use_dhe:
            from .dhe import DHE

            print('Using DHE for homography calculations')
            self.features_matcher = DHE()
            self.features_matcher.load_state_dict(torch.load('patch_stitcher/weights/dhe.pth')['state_dict'])
            # self.features_matcher.load_state_dict(torch.load('patch_stitcher/weights/model_best.pth')['model'])
            # self.features_matcher.load_state_dict(torch.load('patch_stitcher/weights/with_offset/model_best.pth')['model'])
            self.features_matcher.eval()
            self.features_matcher.to(device)
        elif self.use_loftr:
            print('Using LOFTR for homography calculations')
            # self.features_matcher = KF.LoFTR(pretrained='outdoor').to(self.device)
            self.features_matcher = KF.LoFTR(pretrained='indoor').to(self.device)
        elif self.use_sift:
            print('Using SIFT for homography calculations')
            self.features_matcher = cv2.SIFT_create()
            # self.features_matcher = cv2.xfeatures2d.SURF_create()
        else:
            self.features_matcher = None


    def __call__(self, images_paths, is_horizontal=True, is_multidirect=False):
        images = [get_image(image_path) for image_path in images_paths]
        return self.get_stitched_image(images, is_horizontal, is_multidirect)
    

    def get_embeddings(self, data):
        with torch.no_grad():
            data = data.to(self.device) 
            output_embeddings = self.model(data).cpu().numpy() 
        return output_embeddings


    def get_tiles_embeddings(self, data_loader):
        target_embeddings = []
        target_coordinates = []
        
        for tiles_batch, coords_batch in data_loader:
            output_embeddings = self.get_embeddings(tiles_batch)
            target_embeddings.append(output_embeddings)
            target_coordinates.append(coords_batch)

        target_embeddings = np.concatenate(target_embeddings, axis=0)
        target_coordinates = np.concatenate(target_coordinates, axis=0)
        return target_embeddings, target_coordinates


    def get_patches_dataloader(self, image, is_horizontal, is_multidirect):
        source_dataset = TilesDataset(image,
                                      window_size=self.window_size,
                                      step_size=self.step_size,
                                      is_horizontal=is_horizontal,
                                      is_multidirect=is_multidirect,   
                                      transform=self.transform)
        return DataLoader(source_dataset, 
                          batch_size=self.batch_size, 
                          pin_memory=True, 
                          num_workers=self.num_workers)


    def get_patches_dataloader_pyramid(self, image, wind_size, step_size, is_horizontal=True):
        source_dataset = TilesDataset(image,
                                      window_size=wind_size,
                                      step_size=step_size,
                                      is_horizontal=is_horizontal,
                                      is_multidirect=False,   
                                      transform=self.transform)
        return DataLoader(source_dataset, 
                          batch_size=self.batch_size, 
                          pin_memory=True, 
                          num_workers=self.num_workers)


    def get_image_parts(self, s_img, t_img, resize_shape=(200, 480), is_horizontal=True):
        i_w, i_h = resize_shape if is_horizontal else resize_shape[::-1]
        t_h, t_w, t_c = t_img.shape
        offset_x_s = int(t_w/2) if is_horizontal else t_w
        offset_y_s = t_h if is_horizontal else int(t_h/4)

        offset_x = int(t_w/2) if is_horizontal else 0
        offset_y = 0 if is_horizontal else int(t_h/4)

        scale_w = offset_x / i_w if is_horizontal else t_w/i_w
        scale_h = t_h / i_w if is_horizontal else offset_y / i_h
    
        img_s_raw = cv2.resize(s_img[:offset_y_s, :offset_x_s], (i_w, i_h))
        img_t_raw = cv2.resize(t_img[offset_y:, offset_x:], (i_w, i_h))

        # offset_x = 0
        # offset_y = 0
        # scale_w = t_w / i_w
        # scale_h = t_h / i_w

        # img_s_raw = cv2.resize(s_img[:t_w, :t_h], (i_w, i_h))
        # img_t_raw = cv2.resize(t_img, (i_w, i_h))

        img_s_raw = cv2.cvtColor(img_s_raw, cv2.COLOR_BGR2GRAY)
        img_t_raw = cv2.cvtColor(img_t_raw, cv2.COLOR_BGR2GRAY)
    
        return img_s_raw, img_t_raw, (scale_w, scale_h), (offset_x, offset_y)


    def get_homography_dhe(self, s_img, t_img, im_i=0, is_horizontal=True):
        input_shape = (128, 128)
        img_s_raw, img_t_raw, (scale_w, scale_h), (offset_x, offset_y) =self.get_image_parts(s_img, t_img, 
                                                                                         resize_shape=input_shape,  
                                                                                         is_horizontal=is_horizontal)
        
        with torch.no_grad():
            image = np.dstack((img_s_raw, img_t_raw))
            image = torch.from_numpy((image.astype(float)-127.5)/127.5)
            image = image.unsqueeze(0)
            image = image.to(device=self.device)
            image = image.permute(0,3,1,2).float()
            with torch.no_grad():
                output = self.features_matcher(image.float()).cpu().numpy()[0] * 32
        
        s_t_p, s_l_p = (0,0),                            (input_shape[1], 0)
        s_b_p, s_r_p = (input_shape[1], input_shape[0]), (0, input_shape[0])

        four_points = np.array([*s_t_p, *s_l_p, *s_b_p, *s_r_p])
        t_p = four_points + output

        four_points_s = [s_t_p, s_l_p, s_b_p, s_r_p]
        four_points_t = t_p.reshape(-1, 2)

        four_points_s = [(scale_w*x, scale_h*y) for x,y in four_points_s]
        four_points_t = [(scale_w*x+offset_x, scale_h*y+offset_y) for x,y in four_points_t]
      
        M = cv2.getPerspectiveTransform(np.float32(four_points_t), np.float32(four_points_s))

        return M


    def get_homography_loftr(self, s_img, t_img, im_i=0, is_horizontal=True):
        img_s_raw, img_t_raw, (scale_w, scale_h), (offset_x, offset_y) =self.get_image_parts(s_img, t_img, 
                                                                                         resize_shape=(200, 480),  
                                                                                         is_horizontal=is_horizontal)
        
        with torch.no_grad():
            matches = self.features_matcher({'image0':load_torch_image(img_s_raw).to(self.device), 
                                             'image1':load_torch_image(img_t_raw).to(self.device)})
        kps_s = matches['keypoints0'].cpu().numpy()
        kps_t = matches['keypoints1'].cpu().numpy()
        confs = matches['confidence'].cpu().numpy()


        conf_mask = confs>self.loftr_conf_thresh
        kps_s = kps_s[conf_mask]
        kps_t = kps_t[conf_mask]
        confs = confs[conf_mask]

        kps_s_vis = kps_s.copy() 
        kps_t_vis = kps_t.copy()

        kps_s = np.multiply(kps_s, np.array([scale_w, scale_h])) 
        kps_t = np.multiply(kps_t, np.array([scale_w, scale_h]))

        if self.use_kp_filtering:
            outliers_mask_dist = find_outliers_distances(kps_s, kps_t, m=2)
            outliers_mask_angl = find_outliers_angle(kps_s, kps_t, m=2)
            outliers_mask = np.logical_and(outliers_mask_dist, outliers_mask_angl)
            kps_s = kps_s[outliers_mask]
            kps_t = kps_t[outliers_mask]
            confs = confs[outliers_mask]
            kps_s_vis = kps_s_vis[outliers_mask]
            kps_t_vis = kps_t_vis[outliers_mask]

        if self.save_steps_vis:
            make_matching_figure(img_s_raw, img_t_raw, kps_s_vis, kps_t_vis, confs, path=self.vis_save_path/f'matching_{im_i}_{im_i+1}.png')

        kps_s -= np.array([offset_x, offset_y])
        
        if len(kps_s) > self.MIN_MATCH_COUNT:
            M, mask = cv2.findHomography(kps_t, kps_s, cv2.RANSAC, 4)
        else:
            print('Not enough matching pairs')
            return None

        return M


    def get_homography_simple(self, s_img, t_img, im_i=0, is_horizontal=True):\

        img_s_raw, img_t_raw, (scale_w, scale_h), (offset_x, offset_y) =self.get_image_parts(s_img, t_img, 
                                                                                             resize_shape=(200, 480),
                                                                                             is_horizontal=is_horizontal)


        kp1, des1 = self.features_matcher.detectAndCompute(img_s_raw,None)
        kp2, des2 = self.features_matcher.detectAndCompute(img_t_raw,None)

        match = cv2.BFMatcher()
        matches = match.knnMatch(des1,des2,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)
        good = np.array(good)
        M=None    

        kps_s = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
        kps_t = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)

        kps_s_vis = kps_s.copy() 
        kps_t_vis = kps_t.copy()

        kps_s = np.multiply(kps_s, np.array([scale_w, scale_h])) 
        kps_t = np.multiply(kps_t, np.array([scale_w, scale_h])) + np.array([offset_x, offset_y])

        if self.use_kp_filtering:
            outliers_mask_dist = find_outliers_distances(kps_s, kps_t, m=2)
            outliers_mask_angl = find_outliers_angle(kps_s, kps_t, m=2)
            
            outliers_mask = np.logical_and( outliers_mask_dist, outliers_mask_angl)
            
            kps_s = kps_s[outliers_mask]
            kps_t = kps_t[outliers_mask]
            kps_s_vis = kps_s_vis[outliers_mask]
            kps_t_vis = kps_t_vis[outliers_mask]
        confs = np.ones_like(kps_s[:,0])  

        if len(kps_s) > self.MIN_MATCH_COUNT:
            M, mask = cv2.findHomography(kps_s, kps_t, cv2.RANSAC, 3.0)
            M = inv(M)

            make_matching_figure(img_s_raw, img_t_raw, kps_s_vis, kps_t_vis, confs, path=self.vis_save_path/f'matching_{im_i}_{im_i+1}.png')
        else:
            print('Not enough matching pairs')
            return M
        
        return M


    def get_stitched_image(self, images, is_horizontal, is_multidirect):
        stitched_image = None
        
        for im_i in tqdm(range(len(images)-1)):

            source_image = images[im_i]
            target_image = images[im_i+1]

            s_img = cv2.resize(source_image, (0,0), fx=self.o_s, fy=self.o_s)
            t_img = cv2.resize(target_image, (0,0), fx=self.o_s, fy=self.o_s)

            s_h, s_w = s_img.shape[:2]
            t_h, t_w = t_img.shape[:2]

            if self.use_dhe:
                M = self.get_homography_dhe(s_img, t_img, im_i, is_horizontal=is_horizontal)
            elif self.use_loftr:
                M = self.get_homography_loftr(s_img, t_img, im_i, is_horizontal=is_horizontal)
            elif self.use_sift:
                M = self.get_homography_simple(s_img, t_img, im_i, is_horizontal=is_horizontal)
            else:
                M = None

            if self.use_metric:
                if M is not None:
                    if is_horizontal:
                        
                        t_img_warped,_,_ = warp_target(s_img, t_img, M)
                        # t_img_warped = cv2.warpPerspective(t_img, M, (t_w+s_w, t_h))
                        t_img_warped_trim = crop(t_img_warped, only_lr=True)
                    else:
                        # t_img_warped = cv2.warpPerspective(t_img, M, (t_w, s_h+t_h))
                        t_img_warped, _,_ = warp_target(s_img, t_img, M)
                        t_img_warped_trim = crop(t_img_warped, only_ud=True)
                    t_img = t_img_warped_trim

                s_img, t_img = make_border(s_img, t_img, is_horizontal)

                if self.pyramid_search:
                    s_img_curr = s_img.copy()
                    t_img_curr = t_img.copy()
                    sx_offset, sy_offset = 0, 0
                    tx_offset, ty_offset = 0, 0

                    wind_sizes = [200]
                    step_sizes = [10]
                    for step in range(len(wind_sizes)):
                        wind_size, step_size = wind_sizes[step], step_sizes[step]
                        source_data_loader = self.get_patches_dataloader_pyramid(s_img_curr, wind_size, step_size, is_horizontal)
                        target_data_loader = self.get_patches_dataloader_pyramid(t_img_curr, wind_size, step_size, is_horizontal)

                        source_embeddings, source_coordinates = self.get_tiles_embeddings(source_data_loader)
                        target_embeddings, target_coordinates = self.get_tiles_embeddings(target_data_loader)

                        cosine_sim_matrix = cosine_similarity(source_embeddings, target_embeddings)
                        max_similar = np.unravel_index(np.argmax(cosine_sim_matrix), cosine_sim_matrix.shape)

                        source_coords_best = source_coordinates[max_similar[0]]
                        target_coords_best = target_coordinates[max_similar[1]]

                        s_coords = xywh2xyxy(source_coords_best) 
                        t_coords = xywh2xyxy(target_coords_best)
                        (xs_1, ys_1), (xs_2, ys_2) = s_coords
                        (xt_1, yt_1), (xt_2, yt_2) = t_coords
                        s_img_curr = s_img_curr[ys_1:ys_2, xs_1:xs_2]
                        t_img_curr = t_img_curr[yt_1:yt_2, xt_1:xt_2]
                        if is_horizontal:
                            sx_offset += xs_1
                            tx_offset += xt_1
                        else:
                            sy_offset += ys_1
                            ty_offset += yt_1
                    s_coords = [(c_x+sx_offset, c_y+sy_offset)for c_x, c_y in s_coords]
                    t_coords = [(c_x+tx_offset, c_y+ty_offset)for c_x, c_y in t_coords]
                elif self.fast_search:
                    constant_offset = 100
                    offset_x_s = self.window_size if is_horizontal else s_w
                    offset_y_s = s_h if is_horizontal else self.window_size
                    offset_xx = constant_offset if is_horizontal else 0
                    offset_yy = 0 if is_horizontal else constant_offset
                    offset_x_s+=offset_xx
                    offset_y_s+=offset_yy

                    s_img_sample = self.transform(image=s_img[offset_yy:offset_y_s, 
                                                              offset_xx:offset_x_s])
                    s_img_input = s_img_sample['image']
                    s_img_input = s_img_input.unsqueeze(0)

                    source_embeddings = self.get_embeddings(s_img_input)

                    target_data_loader = self.get_patches_dataloader(t_img, is_horizontal, False)
                    target_embeddings, target_coordinates = self.get_tiles_embeddings(target_data_loader)

                    cosine_sim_matrix = cosine_similarity(source_embeddings, target_embeddings)
                    max_similar = np.unravel_index(np.argmax(cosine_sim_matrix), cosine_sim_matrix.shape)

                    target_coords_best = target_coordinates[max_similar[1]]

                    s_coords = (offset_xx, offset_yy), (offset_x_s, offset_y_s)
                    t_coords = xywh2xyxy(target_coords_best)
                        
                else:
                    source_data_loader = self.get_patches_dataloader(s_img, is_horizontal, is_multidirect)
                    target_data_loader = self.get_patches_dataloader(t_img, is_horizontal, is_multidirect)

                    source_embeddings, source_coordinates = self.get_tiles_embeddings(source_data_loader)
                    target_embeddings, target_coordinates = self.get_tiles_embeddings(target_data_loader)

                    cosine_sim_matrix = cosine_similarity(source_embeddings, target_embeddings)
                    max_similar = np.unravel_index(np.argmax(cosine_sim_matrix), cosine_sim_matrix.shape)

                    source_coords_best = source_coordinates[max_similar[0]]
                    target_coords_best = target_coordinates[max_similar[1]]

                    s_coords = xywh2xyxy(source_coords_best) 
                    t_coords = xywh2xyxy(target_coords_best)

                if self.save_steps_vis:
                    source_image_vis = draw_visualization(s_img, s_coords, color=(0,255,0))
                    target_image_vis = draw_visualization(t_img, t_coords, color=(255,0,0))
                    show_images([target_image_vis, source_image_vis], n_col=2, save_name=self.vis_save_path/f'coords_{im_i}_{im_i+1}.png')
                    plot_tiles_similarity(cosine_sim_matrix, save_path=self.vis_save_path/f'{im_i}.png')
                
                if is_horizontal:
                    if im_i==0:
                        stitched_image = np.concatenate([t_img[:, :t_coords[1][0],:], s_img[:, s_coords[1][0]:, :]], axis=1)
                    else:
                        stitched_image, t_img = make_border(stitched_image, t_img, is_horizontal)
                        stitched_image = np.concatenate([t_img[:, :t_coords[1][0],:], stitched_image[:, s_coords[1][0]:, :]], axis=1)
                else:
                    if im_i==0:
                        stitched_image = np.concatenate([t_img[:t_coords[1][1], :,:], s_img[s_coords[1][1]:, :, :]], axis=0)
                    else:
                        stitched_image, t_img = make_border(stitched_image, t_img, is_horizontal)
                        stitched_image = np.concatenate([t_img[:t_coords[1][1], :,:], stitched_image[s_coords[1][1]:, :, :]], axis=0)
            else:
                if M is not None:
                    if im_i==0:
                        stitched_image = s_img
                    stitched_image = stitch_two_images(stitched_image, t_img, M)
            
            show_images([t_img, s_img, stitched_image], n_col=3, save_name=self.vis_save_path/f'warped_{im_i}_{im_i+1}.png')

            
        if stitched_image is not None:
            stitched_image = cv2.cvtColor(crop(stitched_image), cv2.COLOR_BGR2RGB) 
        return stitched_image



