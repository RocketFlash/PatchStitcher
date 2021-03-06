import os
import cv2
import argparse
from pathlib import Path
from os.path import isfile
from patch_stitcher.utils import get_device
from patch_stitcher.patch_stitcher_simple import PatchStitcher
import json
import shutil


def main(source, stitcher, is_horizontal=True, is_multidirect=False, input_data=None, images_folder=None, save_name='stitched.png', save_path='./'):
                         
    if images_folder is None:
        images_paths = sorted([l for l in list(source.glob('*.jpeg')) + \
                                        list(source.glob('*.jpg')) + \
                                        list(source.glob('*.png'))])
    else:
        images_paths = [images_folder / Path(x).name for x in input_data['frame_paths']]
    

    stitched_image = stitcher(images_paths, is_horizontal, is_multidirect)
    cv2.imwrite(str(save_path/f'{save_name}'), stitched_image)
    shutil.copy(save_path/f'{save_name}', stitcher.vis_save_path)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='path to the pretrained model')
    parser.add_argument('--source', type=str, default='', help='images folder path')
    parser.add_argument('--window_size', type=int, default=500, help='tiles window size')
    parser.add_argument('--step_size', type=int, default=10, help='tiles step size')
    parser.add_argument('--batch_size', type=int, default=50, help='embeddings network batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--output_scale', type=float, default=0.5, help='output images scale')
    parser.add_argument('--device', type=str, default='', help='device')
    parser.add_argument('--multidirect', action='store_true', help='use multidirection search')
    parser.add_argument('--save_vis', type=bool, default=True, help='save stitcher steps visualization')
    parser.add_argument('--pyramid_search', action='store_true', help='use pyramid search')
    parser.add_argument('--fast_search', action='store_true', help='use fast search')
    parser.add_argument('--use_dht', action='store_true', help='use dht')
    parser.add_argument('--exp_name', type=str, default='exp1', help='name of experiment')
    parser.add_argument('--tmp', default="./results", help='tmp')
    args = parser.parse_args()

    source = Path(args.source)

    if args.model is not None:
        if isfile(args.model):
            model_path = args.model
        else:
            print("no pretrained model found at '{}'".format(args.model))
            raise ValueError
    else:
        model_path = None

    if args.tmp != "":
        save_path = Path(args.tmp)
        save_path.mkdir(exist_ok=True)

    device = get_device(args.device)

    print(f'Device: {device}')

    scan_idx = source.stem
    exp_path = save_path / args.exp_name
    save_path_scan = exp_path /scan_idx
    save_path_stitched = save_path_scan / 'stitched'
    save_path_stitched.mkdir(exist_ok=True, parents=True)

    stitcher = PatchStitcher(model_path,
                             device=device,
                             output_images_scale=args.output_scale,
                             window_size=args.window_size,
                             step_size=args.step_size,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             vis_save_path=args.tmp,
                             pyramid_search=args.pyramid_search,
                             fast_search=args.fast_search,
                             save_vis=args.save_vis,
                             use_dht=args.use_dht)

    
    for call_idx in range(4):

        call_folder = source / f'call_{call_idx}'
        with open(call_folder / 'stitching_input.json') as fio:
            input_data = json.load(fio)

        images_path = call_folder / 'frames'
        save_path_scan_call = save_path_scan / f'call_{call_idx}'
        save_path_scan_call.mkdir(exist_ok=True, parents=True)
        stitcher.vis_save_path = save_path_scan_call
        
        main(images_path, stitcher, is_horizontal=True, 
                                    is_multidirect=args.multidirect, 
                                    input_data=input_data, 
                                    images_folder=images_path, 
                                    save_name=f'stitched_{call_idx}.png', 
                                    save_path=save_path_stitched)

    save_path_scan_call = save_path_scan / f'call_final'
    save_path_scan_call.mkdir(exist_ok=True, parents=True)
    stitcher.vis_save_path = save_path_scan_call
    main(save_path_stitched, stitcher, is_horizontal=False, 
                                       is_multidirect=args.multidirect, 
                                       save_name=f'stitched_full.png', 
                                       save_path=save_path_scan)
    shutil.copy(save_path_scan/f'stitched_full.png', exp_path/f'stitched_{scan_idx}.png')