## Code from:
# https://github.com/biubug6/Pytorch_Retinaface/detect.py
#
# Changes/additions by: Alexander Hustinx
# Runs all images within the given directory through the detection-operation twice:
#    1) To correct orientation
#    2) To get the detected bounding box


from __future__ import print_function

from glob import glob

import os
from os.path import splitext
from os import listdir
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from matplotlib import pyplot as plt

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import torchvision.transforms.functional as TF
import time
import PIL

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

# Additions
parser.add_argument('--multiple_per_image', action="store_true", default=False,
                    help='Use when image contains multiple faces (Not recommended...)')
parser.add_argument('--images_dir', default="./examples/", help='Location of the images to run detector on')
parser.add_argument('--save_dir', default="./images_cropped/", help='Where the detections are saved')
parser.add_argument('--save_dir_verbose', default="./examples_verbose/", help='Where the verbose images are saved')
parser.add_argument('--save_image_verbose', action="store_true", default=False,
                    help='save intermediate detection results')
parser.add_argument('--use_subdirectories', action="store_true", default=False,
                    help='When set saves images in dirs in the "images_dir" (e.g. for CASIA)')
parser.add_argument('--check_orientation', action="store_true", default=False,
                    help='When set checks the orientation of the detected face, skipping it if too extreme')
parser.add_argument('--crop_size', type=int, default=0,
                    help='Desired width and height of the resulting cropped face, if 0 crops to actual bounding box (default = 0)')
parser.add_argument('--result_type', default='crop', type=str,
                    help="Desired result type from pipeline, options: \'crop\' and \'coords\' (default: \'crop\')")
parser.add_argument('--fill_color', type=float, default=0.5,
                    help="Color (float) that will be used to fill in the expanded regions after rotation (default: 0.5)")

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def correct_orientation(img, landmarks):
    # removed unnormalization, might not be needed ...
    origin = landmarks[[5, 6]]
    middle_finger = landmarks[[7, 8]]
    orientation_vector = middle_finger - origin
    destination_vector = np.array([1., 0.])
    dir_unit_vector = orientation_vector / np.linalg.norm(orientation_vector)
    angle_rad = np.arccos(np.clip(np.dot(destination_vector, dir_unit_vector), -1.0, 1.0))
    angle_degrees = -180 / np.pi * angle_rad
    angle_degrees = angle_degrees if (orientation_vector[1] < 0) else -angle_degrees
    corr_img = TF.rotate(TF.to_pil_image(img), angle=angle_degrees,
                         resample=PIL.Image.BILINEAR)  # , center=list(origin.numpy()))


    # correct landmarks too
    corr_landmarks = landmarks
    return corr_img, angle_degrees


def rotate_image(image, landmarks):
    # define origin
    origin = landmarks[[5, 6]]
    middle_finger = landmarks[[7, 8]]

    nose = landmarks[[9, 10]]

    # calc angle to rotate
    orientation_vector = middle_finger - origin
    destination_vector = np.array([1., 0.])
    dir_unit_vector = orientation_vector / np.linalg.norm(orientation_vector)
    angle_rad = np.arccos(np.clip(np.dot(destination_vector, dir_unit_vector), -1.0, 1.0))
    angle_degrees = -180 / np.pi * angle_rad
    angle_degrees = angle_degrees if (orientation_vector[1] < 0) else -angle_degrees

    # rot_mat = cv2.getRotationMatrix2D(tuple(np.array(image.shape[1::-1]) / 2), angle_degrees, 1.0)
    rot_mat = cv2.getRotationMatrix2D(tuple(nose), angle_degrees, 1.0)

    # Background fill color is set to 0.5 to 0 when centered around [-1,1]
    fill = int(args.fill_color * 255)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(fill,fill,fill))

    # plt.imshow(result)
    # plt.show()

    return result, angle_degrees


def resize_square_aspect(img, desired_size=100):
    old_size = img.size  # (width, height)

    # we crop without resize if desired_size == 0
    if desired_size == 0:
        desired_size = max(old_size)

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = img.resize(new_size, PIL.Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_img = PIL.Image.new("RGB", (desired_size, desired_size))
    new_img.paste(im, ((desired_size - new_size[0]) // 2,
                       (desired_size - new_size[1]) // 2))

    return new_img


# TODO: try/catch OOM-error?
# TODO: This doesn't work as the images are too big (trained on 640x640), so we should shrink the image first
#       get the locations and rescale them to the original size, finally crop to those dimensions of the original image
def resize_square_aspect_cv2(img, desired_size=640):
    old_size = img.shape[0:2]  # (width, height)

    # we crop without resize if desired_size == 0
    if desired_size == 0:
        desired_size = max(old_size)

        # Too large images can cause an OOM-error, hopefully this addresses that...
        if (old_size[0]*old_size[1]) > 10000000:
            desired_size = 2000     # should be large enough

    ratio = float(desired_size) / max(old_size)
    new_size = [int(x * ratio) for x in old_size]
    new_size = tuple(new_size[::-1])

    new_img = cv2.resize(img, new_size)

    return new_img


# Quick and dirty method to determine if a face is turned too much to the side (silhouette)
def is_correct_orientation(landmarks):
    eye_left_x = landmarks[5]
    eye_right_x = landmarks[7]
    # mouth_left = landmarks[[11, 12]]
    # mouth_right = landmarks[[13, 14]]

    box_center_x = landmarks[0] + (landmarks[2] - landmarks[0]) / 2

    # If both eyes are on the same side of the bounding box x-axis (landmarks[[0,2]])
    if (eye_left_x > box_center_x and eye_right_x > box_center_x) or \
            (eye_left_x < box_center_x and eye_right_x < box_center_x):
        return False

    return True


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1
    if args.images_dir == "./examples/":
        print(f"--images_dir not set, using default: {args.images_dir}")
    source_dir = args.images_dir

    if args.result_type != 'crop' and args.result_type != 'coords':
        print(f"Invalid result type requested! (args.result_type: {args.result_type} given)")
        exit()

    ## info logging
    print(f"Using images_dir: {args.images_dir}")
    print(f"Using save_dir: {args.save_dir}")
    print(f"Saving verbose images: {args.save_image_verbose}")
    if args.save_image_verbose:
        print(f"Using save_dir_verbose: {args.save_dir_verbose}")
    print(f"Allowing multiple faces per image: {args.multiple_per_image}")
    print(f"Using subdirectories: {args.use_subdirectories}")
    print(f"Desired result type: {args.result_type}")
    ##

    coords_file = None
    if args.result_type == 'coords':
        os.makedirs(f"{args.save_dir}", exist_ok=True)
        coords_file = open(f"{args.save_dir}_face_coords.csv", 'w+')  # open file in override mode
        coords_file.write('img,bbox_x1,bbox_y1,bbox_x2,bbox_y2,kp1_x,kp1_y,kp2_x,kp2_y,kp3_x,kp3_y,kp4_x,kp4_y,kp5_x,kp5_y\n')
        print(f"Created \'{args.save_dir}_face_coords.csv\' file to store the bounding box coords of the detected faces (on rotated images)")
    img_names = [y for x in os.walk(source_dir) for y in glob(os.path.join(x[0], '*.*'))]

    # List containing the file names of all images that were skipped due to incorrect face orientation
    missed_images = []

    # # If model freezes somewhere halfway.. use this snippet to continue from that point
    # start_idx = -1
    # for idx in range(len(img_names)):
    #     if '\\6642.' in img_names[idx]:   # index where it froze
    #         start_idx = idx
    #         break
    # print(start_idx)
    # img_names = img_names[start_idx-1::]

    for img_path in img_names:
        save_dir = f"{args.save_dir}"
        save_dir_verbose = f"{args.save_dir_verbose}"

        subdir_name = ''
        if args.use_subdirectories:
            subdir_name = (img_path.split('/')[-1]).split('\\')[0]
            save_dir = f"{save_dir}/{subdir_name}"
            save_dir_verbose = f"{save_dir_verbose}/{subdir_name}"
            print(save_dir)

        os.makedirs(f"{save_dir}", exist_ok=True)
        if args.save_image_verbose:
            os.makedirs(f"{save_dir_verbose}", exist_ok=True)

        img_name = (img_path.split('\\')[-1]).split('.')[0]

        # We'll run the detection twice:
        #    first: Correct the image rotation
        #    second: Get the detection
        first = True
        for i in range(2):
            if first:
                print(img_path)
                # *.gif format is not supported by cv.imread(..)
                if img_path.split('.')[-1] == "gif":
                    cap = cv2.VideoCapture(img_path)
                    ret, img_raw_original = cap.read()
                    cap.release()
                else:
                    img_raw_original = cv2.imread(img_path)

                # Note: Be aware of possible size increase (followed by decrease) that can damage the image quality
                original_size = img_raw_original.shape[0:2]
                img_raw = resize_square_aspect_cv2(img_raw_original, 640) #Note: some images are too big resulting in an OOM-error
            else:
                # Use the rotated image as input for the detector
                img_raw = img_rot
                img_raw_original = img_original_rot

            img = np.float32(img_raw)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            tic = time.time()
            with torch.no_grad():
                loc, conf, landms = net(img)  # forward pass
            # print('net forward time on {}: {:.4f}'.format(img_path, time.time() - tic))

            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)

            # Own addition to keep only the single most confident detection (we want single faces)
            if not args.multiple_per_image:
                if len(dets) == 0:
                    first = True
                    continue
                dets = [dets[dets[:, 4].argmax()]]

            # show image
            if args.save_image:
                for idx, b in enumerate(dets):
                    # scale b back to original input size
                    b_scaled = b * float(max(original_size)) / max(img_raw.shape[0:2])

                    # Rotate the image to have the most confident detection's orientation corrected
                    if first:
                        img_rot, rotation_angle_first = rotate_image(img_raw, b)
                        img_original_rot, _ = rotate_image(img_raw_original, b_scaled)
                        continue

                    # Check if the orientation of the face is correct: i.e. rear or frontal facing
                    if args.check_orientation:
                        if not is_correct_orientation(b):
                            missed_images.append(img_name)
                            continue

                    if args.result_type == 'coords':
                        #d = list(map(int, b))
                        d = list(map(int, b_scaled))
                        coords_file.write(f"{f'{subdir_name}/' if args.use_subdirectories else ''}{img_name}_{str(idx) + '_' if args.multiple_per_image else ''}rot.jpg,"
                                          f"{str(d[0:4]+d[5::]).replace('[','').replace(']','').replace(' ','')}\n")

                    # Attempt to crop relevant face box
                    img_rot = TF.to_pil_image(img_rot)
                    img_crop = img_rot.crop((b[0], b[1], b[2], b[3]))

                    img_original_rot = TF.to_pil_image(img_original_rot)
                    img_original_crop = img_original_rot.crop((b_scaled[0], b_scaled[1], b_scaled[2], b_scaled[3]))

                    # create squared rotated crop (if crop_size == 0, we don't resize)
                    if args.crop_size != 0:
                        img_square_crop = img_crop.resize((args.crop_size, args.crop_size))
                    else:
                        img_square_crop = img_original_crop
                    img_square_crop = np.ascontiguousarray(img_square_crop)

                    if args.result_type == 'crop':
                        cv2.imwrite(
                            f"{save_dir}/{img_name}_{str(idx) + '_' if args.multiple_per_image else ''}crop_square.jpg",
                            img_square_crop)
                    elif args.result_type == 'coords' and args.save_image_verbose:
                        cv2.imwrite(
                            f"{save_dir_verbose}/{img_name}_{str(idx) + '_' if args.multiple_per_image else ''}crop_square.jpg",
                            img_square_crop)

                    # create (aspect ratio) squared rotated crop padded with black pixels
                    img_square_padded_crop = resize_square_aspect(img_crop, args.crop_size)

                    # Save rotated image (not verbose when it's the desired result type)
                    img_rot = np.ascontiguousarray(img_original_rot)
                    if args.result_type == 'coords':
                        cv2.imwrite(
                            f"{save_dir}/{img_name}_{str(idx) + '_' if args.multiple_per_image else ''}rot.jpg",
                            img_rot)
                    # Save intermediate verbose images (rots, crops, boxes, etc.)
                    if args.save_image_verbose:
                        # save rotated detection

                        if args.result_type == 'crop':
                            cv2.imwrite(
                                f"{save_dir_verbose}/{img_name}_{str(idx) + '_' if args.multiple_per_image else ''}rot.jpg",
                                img_rot)

                        # save square cropped and padded detection
                        img_square_padded_crop = np.ascontiguousarray(img_square_padded_crop)
                        cv2.imwrite(
                            f"{save_dir_verbose}/{img_name}_{str(idx) + '_' if args.multiple_per_image else ''}"
                            f"crop_square_padded.jpg", img_square_padded_crop)

                        # save crop
                        img_crop = np.ascontiguousarray(img_crop)
                        cv2.imwrite(
                            f"{save_dir_verbose}/{img_name}_{str(idx) + '_' if args.multiple_per_image else ''}"
                            f"crop.jpg", img_crop)


                        img_raw = np.ascontiguousarray(img_raw)

                        text = "{:.4f}".format(b[4])
                        b = list(map(int, b))
                        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                        cx = b[0]
                        cy = b[1] + 12
                        cv2.putText(img_raw, text, (cx, cy),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                        # landms
                        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

                        # plt.close()
                        # plt.imshow(img_raw)
                        # plt.waitforbuttonpress()
                        # exit()

                # save verbose image?
                if args.save_image_verbose:
                    cv2.imwrite(f"{save_dir_verbose}/{img_name}{'_' + str(idx) if args.multiple_per_image else ''}.jpg"
                                , img_raw)

            first = not first

    if args.result_type == 'coords':
        coords_file.flush()
        coords_file.close()

    print(f"{missed_images=}")
