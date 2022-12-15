# Script that aligns faces in images, with provided 5 landmarks, and crops to 112x112
# Uses snippets from a repo of insightface (https://github.com/deepinsight/insightface)

import os
import argparse
from glob import glob

import cv2
import numpy as np
import pandas as pd
from skimage import transform as trans

# command line arguments
parser = argparse.ArgumentParser(description='Face aligner')
parser.add_argument('--images_dir', default="./examples/", help='Location of the images to run detector on')
parser.add_argument('--coords_file', default="./examples.csv", help='Location of the file containing the face coords')
parser.add_argument('--save_dir', default="./images_cropped/", help='Where the detections are saved')
parser.add_argument('--save_dir_verbose', default="./examples_verbose/", help='Where the verbose images are saved')
parser.add_argument('--save_image_verbose', action="store_true", default=False,
                    help='save intermediate detection results')
parser.add_argument('--use_subdirectories', action="store_true", default=False,
                    help='When set saves images in dirs in the "images_dir" (e.g. for CASIA)')
args = parser.parse_args()

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
arcface_src = np.expand_dims(arcface_src, axis=0)


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')

    # Account for the standard image size of ArcFace
    if image_size == 112:
        src = arcface_src
    else:
        src = float(image_size) / 112 * arcface_src

    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

    return min_M, min_index


def norm_crop(img, landmark, image_size=112):
    M, pose_index = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


if __name__ == '__main__':

    resize = 1
    if args.images_dir == "./examples/":
        print(f"--images_dir not set, using default: {args.images_dir}")
    source_dir = args.images_dir

    ## info logging
    print(f"Using images_dir: {args.images_dir}")
    print(f"Using save_dir: {args.save_dir}")
    print(f"Saving verbose images: {args.save_image_verbose}")
    if args.save_image_verbose:
        print(f"Using save_dir_verbose: {args.save_dir_verbose}")
    print(f"Using subdirectories: {args.use_subdirectories}")
    ##

    # Read coords file and get image names
    coords_file = pd.read_csv(args.coords_file)
    img_names = [y for x in os.walk(source_dir) for y in glob(os.path.join(x[0], '*.*'))]

    # attempt to remove <save_dir>/<save_file> from the list of image names to loop through
    for idx, i in enumerate(img_names):
        if ".csv" in i:
            img_names.pop(idx)

    # List containing the file names of all images that were skipped due to incorrect face orientation
    missed_images = []

    for img_path in img_names:
        save_dir = args.save_dir
        save_dir_verbose = args.save_dir_verbose

        subdir_name = ''
        if args.use_subdirectories:
            subdir_name = os.path.split(os.path.split(img_path)[-2])[-1]
            save_dir = os.path.join(save_dir, subdir_name)
            save_dir_verbose = os.path.join(save_dir_verbose, subdir_name)
            print(save_dir)

        os.makedirs(save_dir, exist_ok=True)
        if args.save_image_verbose:
            os.makedirs(save_dir_verbose, exist_ok=True)

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        ## Run the alignment for each image

        # load image
        print(img_path)
        # *.gif format is not supported by cv.imread(..)
        if os.path.splitext(img_path) == ".gif":
            cap = cv2.VideoCapture(img_path)
            ret, img = cap.read()
            cap.release()
        else:
            img = cv2.imread(img_path)

        # TODO: This might still need to be changed to os.path.x(..)
        # get coords into (5,2)-shape
        coords = coords_file[coords_file.img == f"{f'{subdir_name}/' if args.use_subdirectories else ''}{img_name}.jpg"].values[0,5::]
        coords = coords.reshape((5,2)).astype(float)

        # norm crop
        aligned_img = norm_crop(img, coords)

        # save image
        cv2.imwrite(os.path.join(save_dir, f"{img_name}_aligned.jpg"), aligned_img)