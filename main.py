import os
import cv2
import argparse
from pathlib import Path
from os.path import isfile
from patch_stitcher.utils import get_device
from patch_stitcher.patch_stitcher import PatchStitcher


def main(source, stitcher, is_horizontal=True, save_path='./'):
                         

    images_paths = sorted([l for l in list(source.glob('*.jpeg')) + \
                                      list(source.glob('*.jpg')) + \
                                      list(source.glob('*.png'))])
    

    stitched_image = stitcher(images_paths, is_horizontal)
    cv2.imwrite(str(save_path/f'stitched.png'), stitched_image)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to the pretrained model')
    parser.add_argument('--source', type=str, default='', help='images folder path')
    parser.add_argument('--emb_input_size', type=int, default=400, help='embeddings network input size')
    parser.add_argument('--window_size', type=int, default=500, help='tiles window size')
    parser.add_argument('--step_size', type=int, default=10, help='tiles step size')
    parser.add_argument('--batch_size', type=int, default=50, help='embeddings network batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--output_scale', type=float, default=0.5, help='output images scale')
    parser.add_argument('--horizontal', action='store_true', help='search horizontally')
    parser.add_argument('--device', type=str, default='', help='device')
    parser.add_argument('--save_vis', type=bool, default=True, help='save stitcher steps visualization')
    parser.add_argument('--use_loftr', action='store_true', help='use loftr for homography calculation')
    parser.add_argument('--loftr_conf_thresh', type=float, default=0, help='loftr confidence threshold')
    parser.add_argument('--tmp', default="./results", help='tmp')
    args = parser.parse_args()

    source = Path(args.source)

    if args.model:
        if isfile(args.model):
            model_path = args.model
        else:
            print("no pretrained model found at '{}'".format(args.model))
            raise ValueError

    if args.tmp != "":
        save_path = Path(args.tmp)
        save_path.mkdir(exist_ok=True)

    device = get_device(args.device)

    print(f'Device: {device}')

    stitcher = PatchStitcher(model_path,
                             device=device,
                             output_images_scale=args.output_scale,
                             input_size=(args.emb_input_size,
                                         args.emb_input_size),
                             window_size=args.window_size,
                             step_size=args.step_size,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             vis_save_path=args.tmp,
                             use_loftr=args.use_loftr,
                             loftr_conf_thresh=args.loftr_conf_thresh,
                             save_vis=args.save_vis)


    main(source, stitcher, is_horizontal=args.horizontal, save_path=save_path)