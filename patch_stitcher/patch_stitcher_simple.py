import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import xywh2xyxy, plot_tiles_similarity
from .data import TilesDataset
from .utils import get_image, show_images

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from .utils import crop, draw_visualization
from .utils import make_border, resize_border
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from numpy.linalg import inv
from torchvision import transforms
from numpy.lib.stride_tricks import sliding_window_view
from .dht.inference import infer, get_model, load_weights

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
                       save_vis=True,
                       pyramid_search=False,
                       fast_search=True,
                       use_dht=False):
        self.device = device

        input_size = [500, 100]

        if model_path is None:
            from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
            from torchvision import transforms

            self.transform_h = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(input_size),
                                        np.float32,
                                        transforms.ToTensor(),
                                        fixed_image_standardization])
            self.transform_v = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(input_size[::-1]),
                                        np.float32,
                                        transforms.ToTensor(),
                                        fixed_image_standardization])
            self.model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        else:
            self.model = torch.jit.load(model_path, map_location=device).to(device).eval()
            self.transform_h = A.Compose([A.Resize(*input_size),
                                        A.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                        ToTensorV2()])
            self.transform_v = A.Compose([A.Resize(*input_size[::-1]),
                                        A.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                        ToTensorV2()])

        self.use_dht = use_dht
        if use_dht:
            self.dht_model = get_model(num_angle=120,
                                  num_rho=120,
                                  backbone='res2net50',
                                  device=self.device,
                                  hough_cuda=False).eval()
            dht_checkpoint = load_weights(self.dht_model, '/home/rauf/trained_weights/shelves_detection/res2net_512x400_x1_2_ver3.pth')
            # dht_transform = A.Compose([ A.Resize(512, 400),
            #                             A.Normalize(mean=[0.485, 0.456, 0.406],
            #                                         std=[0.229, 0.224, 0.225]),
            #                             ToTensorV2()])
    
        self.window_size = window_size
        self.batch_size = batch_size
        self.step_size = step_size
        self.num_workers = num_workers
        self.o_s = output_images_scale
        self.vis_save_path = Path(vis_save_path)
        self.save_steps_vis = save_vis
        self.pyramid_search = pyramid_search
        self.fast_search = fast_search


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
        if not is_horizontal:
            h, w, c = image.shape
            image = image[int(h/3):, :, :]

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



    def get_stitched_image(self, images, is_horizontal, is_multidirect):
        stitched_image = None

        if is_horizontal:
            self.transform = self.transform_h
        else:
            self.transform = self.transform_v

        for im_i in tqdm(range(len(images)-1)):

            source_image = images[im_i]
            target_image = images[im_i+1]


            if self.use_dht and is_horizontal:
                safe_dht_offset = 10
                if im_i==0:
                    s_img = cv2.resize(source_image, (0,0), fx=self.o_s, fy=self.o_s)
                    s_h, s_w = s_img.shape[:2]

                    s_lines = infer(s_img, self.dht_model, input_size=(512, 400), 
                                            threshold=0.1, 
                                            num_angle=120, 
                                            num_rho=120, 
                                            show_time=False,
                                            device=self.device,
                                            with_tta=False,
                                            filter_vertical=True)
                    s_lines = s_lines[s_lines[:, 3] > safe_dht_offset] if len(s_lines)>0 else s_lines
                    s_lines = s_lines[s_lines[:, 1] > safe_dht_offset] if len(s_lines)>0 else s_lines
                    s_lines = s_lines[s_lines[:, 3] < s_h - safe_dht_offset] if len(s_lines)>0 else s_lines
                    s_lines = s_lines[s_lines[:, 1] < s_h - safe_dht_offset] if len(s_lines)>0 else s_lines
                    s_lines = s_lines[s_lines[:, 3].argsort()]

                    s_ap_top, s_ap_bot = [s_lines[0], s_lines[-1]]
                    s_ap_top_x1, s_ap_top_y1, s_ap_top_x2, s_ap_top_y2 = s_ap_top
                    s_ap_bot_x1, s_ap_bot_y1, s_ap_bot_x2, s_ap_bot_y2 = s_ap_bot
                    s_dst_top_y = min(s_ap_top_y1, s_ap_top_y2)
                    s_dst_bot_y = max(s_ap_bot_y1, s_ap_bot_y2)
                    if len(s_lines>1):
                        if s_dst_bot_y-s_dst_top_y>300:
                            s_src = np.array([
                                [s_ap_top_x1, s_ap_top_y1],
                                [s_ap_top_x2, s_ap_top_y2],
                                [s_ap_bot_x2, s_ap_bot_y2],
                                [s_ap_bot_x1, s_ap_bot_y1]], dtype = "float32")

                            s_dst = np.array([
                                [0, s_dst_top_y],
                                [s_w - 1, s_dst_top_y],
                                [s_w - 1, s_dst_bot_y],
                                [0, s_dst_bot_y]], dtype = "float32")
                            
                            M = cv2.getPerspectiveTransform(s_src, s_dst)
                            s_img_ref = cv2.warpPerspective(s_img, M, (s_w, s_h))
                            if self.save_steps_vis:
                                show_images([s_img, s_img_ref], n_col=2, save_name=self.vis_save_path/f's_ref_{im_i}_{im_i+1}.png')
                            s_img = s_img_ref
                else:
                    s_img = t_img
                    s_lines = t_lines
                    s_dst_top_y = t_dst_top_y
                    s_dst_bot_y = t_dst_bot_y

                t_img = cv2.resize(target_image, (0,0), fx=self.o_s, fy=self.o_s)
                t_h, t_w = t_img.shape[:2]

                t_lines = infer(t_img, self.dht_model, input_size=(512, 400), 
                                        threshold=0.1, 
                                        num_angle=120, 
                                        num_rho=120, 
                                        show_time=False,
                                        device=self.device,
                                        with_tta=False,
                                        filter_vertical=True)

                
                t_lines = t_lines[t_lines[:, 3] > safe_dht_offset] if len(t_lines)>0 else t_lines
                t_lines = t_lines[t_lines[:, 1] > safe_dht_offset] if len(t_lines)>0 else t_lines
                t_lines = t_lines[t_lines[:, 3] < t_h - safe_dht_offset] if len(t_lines)>0 else t_lines
                t_lines = t_lines[t_lines[:, 1] < t_h - safe_dht_offset] if len(t_lines)>0 else t_lines
                t_lines = t_lines[t_lines[:, 3].argsort()]

                if self.save_steps_vis:
                    s_img_lines = s_img.copy()
                    t_img_lines = t_img.copy()

                    for line in s_lines:
                        x1, y1, x2, y2 = [int(x) for x in line]
                        cv2.line(s_img_lines, (x1, y1), (x2, y2), (0, 0, 255), 9)

                    for line in t_lines:
                        x1, y1, x2, y2 = [int(x) for x in line]
                        cv2.line(t_img_lines, (x1, y1), (x2, y2), (0, 0, 255), 9)

                    show_images([s_img_lines, t_img_lines], n_col=2, save_name=self.vis_save_path/f'lines_{im_i}_{im_i+1}.png')


                if len(t_lines>1):
                    t_ap_top, t_ap_bot = [t_lines[0], t_lines[-1]]
                    t_ap_top_x1, t_ap_top_y1, t_ap_top_x2, t_ap_top_y2 = t_ap_top
                    t_ap_bot_x1, t_ap_bot_y1, t_ap_bot_x2, t_ap_bot_y2 = t_ap_bot
                    t_dst_top_y = min(t_ap_top_y1, t_ap_top_y2)
                    t_dst_bot_y = max(t_ap_bot_y1, t_ap_bot_y2)

                    if t_dst_bot_y-t_dst_top_y>300:

                        t_src = np.array([
                            [t_ap_top_x1, t_ap_top_y1],
                            [t_ap_top_x2, t_ap_top_y2],
                            [t_ap_bot_x2, t_ap_bot_y2],
                            [t_ap_bot_x1, t_ap_bot_y1]], dtype = "float32")

                        # t_dst = np.array([
                        #     [0, s_dst_top_y],
                        #     [t_w - 1, s_dst_top_y],
                        #     [t_w - 1, s_dst_bot_y],
                        #     [0, s_dst_bot_y]], dtype = "float32")

                        t_dst = np.array([
                            [0, t_dst_top_y],
                            [t_w - 1, t_dst_top_y],
                            [t_w - 1, t_dst_bot_y],
                            [0, t_dst_bot_y]], dtype = "float32")
                        
                        M = cv2.getPerspectiveTransform(t_src, t_dst)
                        t_img_ref = cv2.warpPerspective(t_img, M, (t_w, t_h))
                        if self.save_steps_vis:
                            show_images([t_img, t_img_ref], n_col=2, save_name=self.vis_save_path/f't_ref_{im_i}_{im_i+1}.png')
                        t_img = t_img_ref
            else:
                s_img = cv2.resize(source_image, (0,0), fx=self.o_s, fy=self.o_s)
                t_img = cv2.resize(target_image, (0,0), fx=self.o_s, fy=self.o_s)

                s_h, s_w = s_img.shape[:2]
                t_h, t_w = t_img.shape[:2]

            # s_img, t_img = make_border(s_img, t_img, is_horizontal)

            if self.fast_search:
                constant_offset = 2
                offset_x_s = self.window_size if is_horizontal else s_w
                offset_y_s = s_h if is_horizontal else self.window_size
                offset_xx = constant_offset if is_horizontal else 0
                offset_yy = 0 if is_horizontal else constant_offset
                offset_x_s+=offset_xx
                offset_y_s+=offset_yy

                if isinstance(self.transform, transforms.Compose):
                    s_img_input = self.transform(s_img[offset_yy:offset_y_s, offset_xx:offset_x_s])
                    s_img_input = s_img_input.unsqueeze(0)
                else:
                    s_img_sample = self.transform(image=s_img[offset_yy:offset_y_s, 
                                                                offset_xx:offset_x_s])
                    s_img_input = s_img_sample['image']
                    s_img_input = s_img_input.unsqueeze(0)

                source_embeddings = self.get_embeddings(s_img_input)

                target_data_loader = self.get_patches_dataloader(t_img, is_horizontal, False)
                target_embeddings, target_coordinates = self.get_tiles_embeddings(target_data_loader)

                is_euclidean = False
                if is_euclidean:
                    csm = np.array([np.linalg.norm(source_embeddings-kp_t)for kp_t in target_embeddings])
                    cosine_sim_matrix = np.expand_dims(csm, 0)
                else:
                    cosine_sim_matrix = cosine_similarity(source_embeddings, target_embeddings)
                    csm = cosine_sim_matrix.squeeze()
                sim_vals = sliding_window_view(csm, 5).mean(axis=-1)
                index_shift = int((csm.shape[0] - sim_vals.shape[0])/2)

                max_similar = np.argmax(sim_vals) + index_shift

                target_coords_best = target_coordinates[max_similar]

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

            if not is_horizontal:
                ofst_y = int(t_h/3)
                t_coords[0] = (t_coords[0][0]+ ofst_y, t_coords[0][1] + ofst_y)
                t_coords[1] = (t_coords[1][0]+ ofst_y, t_coords[1][1] + ofst_y)

            if self.save_steps_vis:
                source_image_vis = draw_visualization(s_img, s_coords, color=(0,255,0))
                target_image_vis = draw_visualization(t_img, t_coords, color=(255,0,0))
                show_images([target_image_vis, source_image_vis], n_col=2, save_name=self.vis_save_path/f'coords_{im_i}_{im_i+1}.png')
                plot_tiles_similarity(cosine_sim_matrix, save_path=self.vis_save_path/f'{im_i}.png')
            
            if is_horizontal:
                if im_i==0:
                    s_img, t_img = resize_border(s_img, t_img, is_horizontal)
                    stitched_image = np.concatenate([t_img[:, :t_coords[1][0],:], s_img[:, s_coords[1][0]:, :]], axis=1)
                else:
                    stitched_image, t_img = resize_border(stitched_image, t_img, is_horizontal)
                    stitched_image = np.concatenate([t_img[:, :t_coords[1][0],:], stitched_image[:, s_coords[1][0]:, :]], axis=1)
            else:
                if im_i==0:
                    s_img, t_img = resize_border(s_img, t_img, is_horizontal)
                    stitched_image = np.concatenate([t_img[:t_coords[1][1], :,:], s_img[s_coords[1][1]:, :, :]], axis=0)
                else:
                    stitched_image, t_img = resize_border(stitched_image, t_img, is_horizontal)
                    stitched_image = np.concatenate([t_img[:t_coords[1][1], :,:], stitched_image[s_coords[1][1]:, :, :]], axis=0)
            
            show_images([t_img, s_img, stitched_image], n_col=3, save_name=self.vis_save_path/f'warped_{im_i}_{im_i+1}.png')

            
        if stitched_image is not None:
            stitched_image = cv2.cvtColor(crop(stitched_image), cv2.COLOR_BGR2RGB) 
        return stitched_image



