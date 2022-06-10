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
from .utils import load_torch_image, make_matching_figure, crop

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class PatchStitcher:
    def __init__(self, model_path,
                       device='cpu',
                       output_images_scale=0.5,
                       input_size=(400,400),
                       window_size=500,
                       step_size=20,
                       batch_size=50,
                       num_workers=8,
                       vis_save_path='./',
                       use_loftr=False,
                       loftr_conf_thresh=0.5,
                       use_kp_filtering=True,
                       save_vis=True):
        self.device = device
        
        if model_path is None:
            from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
            from torchvision import transforms

            self.transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(input_size),
                                        np.float32,
                                        transforms.ToTensor(),
                                        fixed_image_standardization])
            self.model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        else:
            self.model = torch.jit.load(model_path, map_location=device).to(device).eval()
            self.transform = A.Compose([A.Resize(*input_size),
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
        if self.use_loftr:
            # self.features_matcher = KF.LoFTR(pretrained='outdoor').to(self.device)
            self.features_matcher = KF.LoFTR(pretrained='indoor').to(self.device)
        else:
            self.features_matcher = cv2.xfeatures2d.SIFT_create()
        self.use_kp_filtering = use_kp_filtering


    def __call__(self, images_paths, is_horizontal=True, is_multidirect=False):
        images = [get_image(image_path) for image_path in images_paths]
        return self.get_stitched_image(images, is_horizontal, is_multidirect)


    def get_tiles_embeddings(self, data_loader):
        target_embeddings = []
        target_coordinates = []
        
        for tiles_batch, coords_batch in data_loader:
            with torch.no_grad():
                tiles_batch = tiles_batch.to(self.device) 
                output_embeddings = self.model(tiles_batch).cpu().numpy() 
                target_embeddings.append(output_embeddings)
                target_coordinates.append(coords_batch)

        target_embeddings = np.concatenate(target_embeddings, axis=0)
        target_coordinates = np.concatenate(target_coordinates, axis=0)
        return target_embeddings, target_coordinates


    def make_border(self, source_image, target_image, is_horizontal):
        s_h, s_w, s_c = source_image.shape
        t_h, t_w, t_c = target_image.shape

        if is_horizontal:
            if s_h != t_h:
                border_top = (max(s_h, t_h)-min(s_h, t_h)) // 2
                border_bot = (max(s_h, t_h)-min(s_h, t_h)) - border_top
                if s_h>t_h:
                    target_image = cv2.copyMakeBorder(target_image, border_top, border_bot, 0, 0, borderType=cv2.BORDER_CONSTANT)
                else:
                    source_image = cv2.copyMakeBorder(source_image, border_top, border_bot, 0, 0, borderType=cv2.BORDER_CONSTANT)

        else:
            if s_w != t_w:
                border_left = (max(s_w, t_w)-min(s_w, t_w)) // 2
                border_right = (max(s_w, t_w)-min(s_w, t_w)) - border_left
                if s_w>t_w:
                    target_image = cv2.copyMakeBorder(target_image, 0, 0, border_left, border_right, borderType=cv2.BORDER_CONSTANT)
                else:
                    source_image = cv2.copyMakeBorder(source_image, 0, 0, border_left, border_right, borderType=cv2.BORDER_CONSTANT)

        return source_image, target_image



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
    

    def draw_visualization(self, image_small, coords, color=(255,0,0)): 
        image = image_small.copy()
        cv2.rectangle(image,coords[0],coords[1], color,5)
        return image


    def find_outliers_angle(self, kps1, kps2, m=1):
        ang1 = np.arctan2(kps1[:, 1], kps1[:, 0])
        ang2 = np.arctan2(kps2[:, 1], kps2[:, 0])

        angle_difference = ang2 - ang1
        angles = ((np.rad2deg(angle_difference % (2 * np.pi)) - 180) % 360) - 180

        outliers_mask = find_outliers(angles, m=m)
        
        return outliers_mask

    
    def find_outliers_distances(self, kps_s, kps_t, m=1):
        distances = [np.linalg.norm(kp_s-kp_t)for kp_s, kp_t in zip(kps_s, kps_t)]
        outliers_mask = find_outliers(distances, m=m)
        return outliers_mask


    def get_homography_loftr(self, s_img, t_img, im_i=0, is_horizontal=True):
        i_w, i_h = (200, 480) if is_horizontal else (480, 200)
        s_h, s_w, s_c = s_img.shape
        t_h, t_w, t_c = t_img.shape
        scale_s_w, scale_s_h = s_w / i_w, s_h / i_h
        scale_t_w, scale_t_h = t_w / i_w, t_h / i_h 

        offset_x_s = int(s_w/2) if is_horizontal else s_w
        offset_y_s = s_h if is_horizontal else int(s_h/4)
        offset_x_t = int(t_w/2) if is_horizontal else 0
        offset_y_t = 0 if is_horizontal else int(t_h/4)

        img_s_raw = cv2.resize(s_img[:offset_y_s, :offset_x_s], (i_w, i_h))
        img_t_raw = cv2.resize(t_img[offset_y_t:, offset_x_t:], (i_w, i_h))

        img_s_raw = cv2.cvtColor(img_s_raw, cv2.COLOR_BGR2GRAY)
        img_t_raw = cv2.cvtColor(img_t_raw, cv2.COLOR_BGR2GRAY)
        
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

        kps_s = np.multiply(kps_s, np.array([scale_s_w, scale_s_h])) + np.array([offset_x_t, offset_y_t])
        kps_t = np.multiply(kps_t, np.array([scale_t_w, scale_t_h])) 

        if self.use_kp_filtering:
            outliers_mask_dist = self.find_outliers_distances(kps_s, kps_t, m=500)
            outliers_mask_angl = self.find_outliers_angle(kps_s, kps_t, m=10)
            outliers_mask = np.logical_and(outliers_mask_dist, outliers_mask_angl)
            kps_s = kps_s[outliers_mask]
            kps_t = kps_t[outliers_mask]
            confs = confs[outliers_mask]
            kps_s_vis = kps_s_vis[outliers_mask]
            kps_t_vis = kps_t_vis[outliers_mask]

        if self.save_steps_vis:
            make_matching_figure(img_s_raw, img_t_raw, kps_s_vis, kps_t_vis, confs, path=self.vis_save_path/f'matching_{im_i}_{im_i+1}.png')

        kps_s -= np.array([offset_x_t, offset_y_t])
        MIN_MATCH_COUNT = 15
        if len(kps_s) > MIN_MATCH_COUNT:
            M, mask = cv2.findHomography(kps_s, kps_t, cv2.RANSAC, 3)
        else:
            print('Not enough matching pairs')
            return None

        return M


    # Use SIFT keypoints
    def get_homography_simple(self, s_img, t_img, im_i=0, is_horizontal=True):
        s_h, s_w, s_c = s_img.shape
        t_h, t_w, t_c = t_img.shape
        i_w, i_h = (200, 480) if is_horizontal else (640, 480)
        scale_s_w, scale_s_h = s_w / i_w, s_h / i_h
        scale_t_w, scale_t_h = t_w / i_w, t_h / i_h 

        offset_x_s = int(s_w/2) if is_horizontal else s_w
        offset_y_s = s_h if is_horizontal else int(s_h/4)
        offset_x_t = int(t_w/2) if is_horizontal else 0
        offset_y_t = 0 if is_horizontal else int(t_h/4)

        img_s_raw = cv2.resize(s_img[:offset_y_s, :offset_x_s], (i_w, i_h))
        img_t_raw = cv2.resize(t_img[offset_y_t:, offset_x_t:], (i_w, i_h))


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

        kps_s = np.multiply(kps_s, np.array([scale_s_w, scale_s_h])) + np.array([offset_x_t, offset_y_t])
        kps_t = np.multiply(kps_t, np.array([scale_t_w, scale_t_h]))

        if self.use_kp_filtering:
            outliers_mask_dist = self.find_outliers_distances(kps_s, kps_t, m=100)
            outliers_mask_angl = self.find_outliers_angle(kps_s, kps_t, m=5)
            
            outliers_mask = np.logical_and( outliers_mask_dist, outliers_mask_angl)
            # outliers_mask = outliers_mask_dist
            kps_s = kps_s[outliers_mask]
            kps_t = kps_t[outliers_mask]
            good = good[outliers_mask]

        if len(kps_s) > 10:
            M, mask = cv2.findHomography(kps_s, kps_t, cv2.RANSAC, 5.0)

            kp1 = [cv2.KeyPoint(x,y,1) for x,y in kps_s]
            kp2 = [cv2.KeyPoint(x,y,1) for x,y in kps_t]

            draw_params = dict(matchColor = (0,255,0), singlePointColor = None, flags = 2)
            img_vis = cv2.drawMatches(img_s_raw, kp1, img_t_raw, kp2, good, None,**draw_params)
            cv2.imwrite(str(self.vis_save_path/f'matching_{im_i}_{im_i+1}.png'), img_vis)
        else:
            print('Not enough matching pairs')
            return M
        
        return M


    def get_stitched_image(self, images, is_horizontal, is_multidirect):
        stitched_image = None

        for im_i in tqdm(range(len(images)-1)):

            source_image = images[im_i]
            target_image = images[im_i+1]

            source_image, target_image = self.make_border(source_image, target_image, is_horizontal)
            s_img = cv2.resize(source_image, (0,0), fx=self.o_s, fy=self.o_s)
            t_img = cv2.resize(target_image, (0,0), fx=self.o_s, fy=self.o_s)

            t_h, t_w, t_c = t_img.shape
            s_h, s_w, s_c = s_img.shape

            M = None
            if self.use_loftr:
                M = self.get_homography_loftr(s_img, t_img, im_i, is_horizontal=is_horizontal)
            else:
                M = self.get_homography_simple(s_img, t_img, im_i, is_horizontal=True)

            
            if M is not None:
                if is_horizontal:
                    s_img_warped = cv2.warpPerspective(s_img, M, (t_w+s_w, t_h))
                    s_img_warped_trim = crop(s_img_warped, only_lr=True)
                else:
                    s_img_warped = cv2.warpPerspective(s_img, M, (t_w, s_h+t_h))
                    s_img_warped_trim = crop(s_img_warped, only_ud=True)
                show_images([t_img, s_img_warped_trim], n_col=2, save_name=self.vis_save_path/f'warped_{im_i}_{im_i+1}.png')
                s_img = s_img_warped_trim

            if im_i==0:
                source_data_loader = self.get_patches_dataloader(s_img, is_horizontal, is_multidirect)
                source_embeddings, source_coordinates = self.get_tiles_embeddings(source_data_loader)
            else:
                source_embeddings, source_coordinates = target_embeddings, target_coordinates
                
            
            target_data_loader = self.get_patches_dataloader(t_img, is_horizontal, is_multidirect)
            target_embeddings, target_coordinates = self.get_tiles_embeddings(target_data_loader)

            cosine_sim_matrix = cosine_similarity(source_embeddings, target_embeddings)
            max_similar = np.unravel_index(np.argmax(cosine_sim_matrix), cosine_sim_matrix.shape)

            source_coords_best = source_coordinates[max_similar[0]]
            target_coords_best = target_coordinates[max_similar[1]]

            s_coords = xywh2xyxy(source_coords_best) 
            t_coords = xywh2xyxy(target_coords_best)

            if self.save_steps_vis:
                source_image_vis = self.draw_visualization(s_img, s_coords, color=(0,255,0))
                target_image_vis = self.draw_visualization(t_img, t_coords, color=(255,0,0))
                show_images([target_image_vis, source_image_vis], n_col=2, save_name=self.vis_save_path/f'coords_{im_i}_{im_i+1}.png')
                plot_tiles_similarity(cosine_sim_matrix, save_path=self.vis_save_path/f'{im_i}.png')
            
            if is_horizontal:
                if im_i==0:
                    stitched_image = np.concatenate([t_img[:, :t_coords[1][0],:], s_img[:, s_coords[1][0]:, :]], axis=1)
                else:
                    st_h, st_w, st_c = stitched_image.shape
                    stitched_image = cv2.resize(stitched_image, (st_w, t_img.shape[0]))
                    stitched_image = np.concatenate([t_img[:, :t_coords[1][0],:], stitched_image[:, s_coords[1][0]:, :]], axis=1)
            else:
                if im_i==0:
                    stitched_image = np.concatenate([t_img[:t_coords[1][1], :,:], s_img[s_coords[1][1]:, :, :]], axis=0)
                else:
                    st_h, st_w, st_c = stitched_image.shape
                    stitched_image = cv2.resize(stitched_image, (t_img.shape[1], st_h))
                    stitched_image = np.concatenate([t_img[:t_coords[1][1], :,:], stitched_image[s_coords[1][1]:, :, :]], axis=0)

        if stitched_image is not None:
            stitched_image = cv2.cvtColor(crop(stitched_image), cv2.COLOR_BGR2RGB) 
        return stitched_image



