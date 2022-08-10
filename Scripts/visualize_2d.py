#!/usr/bin/env python
# Author : Maximilian Gilles, IFL, KIT
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
import cv2
import copy
from skimage import io
from skimage import color
from skimage import segmentation
from skimage import measure
import random
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import visualize_layout

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset Obejct detection viewer.")
    parser.add_argument(
        "--data_root",
        type=str, 
        default="./dataset_real/level_5",
        help="Path to data.")
    parser.add_argument(
        "--scene",
        type=int, default=0,
        help="Specify which scene to load.")
    parser.add_argument(
        "--viewpt",
        type=int, default=0,
        help="Specify which viwpt to load.")
    parser.add_argument(
        "--visualize_layout",
        default=False, action='store_true',
        help="Set flag to vis scene labels.")
    parser.add_argument(
        "--real_data",
        default=False, action='store_true',
        help="Set flag for real world data.")
    parser.add_argument(
        "--real_data_grasps",
        default=False, action='store_true',
        help="Set flag for real world data grasps.")
    

    args = parser.parse_args()

    PATH_TO_DATA = pathlib.Path(args.data_root)
    SCENE = args.scene
    VIEWPT = args.viewpt
    PATH_TO_SCENE = PATH_TO_DATA / f"scene{SCENE}"
    PATH_TO_RGB = PATH_TO_SCENE / f"{VIEWPT}_rgb.png"
    PATH_TO_DEPTH = PATH_TO_SCENE / f"{VIEWPT}.npz"
    PATH_TO_GRASPS = PATH_TO_SCENE / f"{VIEWPT}_grasps.npz"

    color_bgr = cv2.imread(str(PATH_TO_RGB))
    color_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    with np.load(str(PATH_TO_DEPTH)) as data:
        depth = data['depth']
        instances_objects = data['instances_objects']
        instances_semantic = data['instances_semantic']
        if args.real_data is False:
            occlusion = data['occlusion']
            try:
                occlusion_masks = data['occlusion_objects']
            except:
                pass
            try:
                occlusion_masks = data['seg_masks_single']
            except:
                pass


    if args.real_data_grasps and args.real_data:
        with np.load(str(PATH_TO_GRASPS)) as data:
            suction_bitmap = data['suction_bitmap']
            parallel_jaw_2d = data['parallel_jaw_2d']

        suction_mask = np.zeros_like(color_img)
        for i in range(suction_mask.shape[0]):
            for j in range(suction_mask.shape[1]):
                if suction_bitmap[i,j] != 0:
                    suction_mask[i,j] = (255,0,0)
                else:
                    suction_mask[i,j] = color_img[i,j]

        parallel_jaw_mask = copy.deepcopy(color_img)
        for pt_pair in parallel_jaw_2d:
            parallel_jaw_mask = cv2.line(parallel_jaw_mask, pt_pair[0], pt_pair[1], (0,0,255), thickness=10)


    else:
        suction_mask = np.zeros_like(color_img)
        parallel_jaw_mask = np.zeros_like(color_img)

    if args.real_data is False:
        # no amodal mask available for real data

        # Segment image with SLIC - Simple Linear Iterative Clustering
        seg = segmentation.slic(color_img, n_segments=30, compactness=40.0, enforce_connectivity=True, sigma=3)
        ## generate seg from occlusion masks
        seg = np.zeros(color_img.shape[:2], dtype=np.uint8)
        mask = instances_objects
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):
                if mask[i,j] != 0:
                    #print(mask[i,j])
                    seg[i,j] = int(mask[i,j])
                else:
                    #print(mask[i,j])
                    pass


        colors=['black', 'dimgray', 'dimgrey', 'gray', 'grey', 'darkgray', 'darkgrey', 'silver', 'lightgray', 'lightgrey',
                'gainsboro', 'whitesmoke', 'white', 'snow', 'rosybrown', 'lightcoral', 'indianred', 'brown', 'firebrick',
                'maroon', 'darkred', 'red', 'mistyrose', 'salmon', 'tomato', 'darksalmon', 'coral', 'orangered', 'lightsalmon',
                'sienna', 'seashell', 'chocolate', 'saddlebrown', 'sandybrown', 'peachpuff', 'peru', 'linen', 'bisque',
                'darkorange', 'burlywood', 'antiquewhite', 'tan', 'navajowhite', 'blanchedalmond', 'papayawhip', 'moccasin',
                'orange', 'wheat', 'oldlace', 'floralwhite', 'darkgoldenrod', 'goldenrod', 'cornsilk', 'gold', 'lemonchiffon',
                'khaki', 'palegoldenrod', 'darkkhaki', 'ivory', 'beige', 'lightyellow', 'lightgoldenrodyellow', 'olive',
                'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'greenyellow', 'chartreuse', 'lawngreen', 'honeydew',
                'darkseagreen', 'palegreen', 'lightgreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen',
                'mediumseagreen', 'springgreen', 'mintcream', 'mediumspringgreen', 'mediumaquamarine', 'aquamarine', 'turquoise',
                'lightseagreen', 'mediumturquoise', 'azure', 'lightcyan', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan',
                'aqua', 'cyan', 'darkturquoise', 'cadetblue', 'powderblue', 'lightblue', 'deepskyblue', 'skyblue', 'lightskyblue',
                'steelblue', 'aliceblue', 'dodgerblue', 'lightslategray', 'lightslategrey', 'slategray', 'slategrey',
                'lightsteelblue', 'cornflowerblue', 'royalblue', 'ghostwhite', 'lavender', 'midnightblue', 'navy', 'darkblue',
                'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'blueviolet', 'indigo',
                'darkorchid', 'darkviolet', 'mediumorchid', 'thistle', 'plum', 'violet', 'purple', 'darkmagenta', 'fuchsia',
                'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink', 'lavenderblush', 'palevioletred', 'crimson',
                'pink', 'lightpink']
        random.shuffle(colors)

        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(color.label2rgb(seg,color_img, saturation=0.4, colors=colors, kind='overlay'))

        #print(occlusion_masks.shape[0])
        for idx in range(occlusion_masks.shape[0]):
            mask = occlusion_masks[int(idx)]
            contours_ = measure.find_contours(mask, 0.8)
            for contour in contours_:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color=colors[idx])


        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('amodal segmentation')
        plt.show()

        height, width = depth.shape
        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.title('RGB')
        plt.imshow(color_img)
        plt.subplot(2, 3, 2)
        plt.axis('off')
        plt.title('Depth')
        plt.imshow(depth)
        plt.subplot(2, 3, 3)
        plt.axis('off')
        plt.title('Instances (Object Ids)')
        plt.imshow(instances_objects)
        plt.subplot(2, 3, 4)
        plt.axis('off')
        plt.title('Instances (Categories)')
        plt.imshow(instances_semantic)
        plt.subplot(2, 3, 5)
        plt.title('Occlusion')
        plt.imshow(occlusion*100, cmap=plt.get_cmap('YlGnBu'))
        plt.axis('off')
        plt.colorbar()
        plt.show()
    
    else:
        height, width = depth.shape
        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.title('RGB')
        plt.imshow(color_img)
        plt.subplot(2, 3, 2)
        plt.axis('off')
        plt.title('Depth')
        plt.imshow(depth, cmap='Greys')
        plt.colorbar()
        plt.subplot(2, 3, 3)
        plt.axis('off')
        plt.title('Instances (Object Ids)')
        plt.imshow(instances_objects)
        plt.subplot(2, 3, 4)
        plt.axis('off')
        plt.title('Instances (Categories)')
        plt.imshow(instances_semantic)
        plt.subplot(2, 3, 5)
        plt.title('Vacuum')
        plt.imshow(suction_mask)
        plt.axis('off')
        plt.subplot(2, 3, 6)
        plt.title('Parallel-Jaw')
        plt.imshow(parallel_jaw_mask)
        plt.axis('off')
        plt.show()

    if args.visualize_layout:
        PATH_TO_RGB = PATH_TO_SCENE / f"3"
        visualize_layout.visualize(str(PATH_TO_RGB))
