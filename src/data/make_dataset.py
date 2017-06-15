import os
import cv2
import h5py
import parmap
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm as tqdm
import matplotlib.pylab as plt


def format_image(img_path, size):
    bgr_img = cv2.imread(img_path)
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Lab)
    lab_img = cv2.resize(lab_img, (size, size), interpolation=cv2.INTER_AREA)

    ab_img = lab_img[:, :, 1:]
    l_img = lab_img[:, :, 0]
    l_img = np.expand_dims(l_img, -1)

    ab_img = np.expand_dims(ab_img, 0).transpose(0, 3, 1, 2)
    l_img = np.expand_dims(l_img, 0).transpose(0, 3, 1, 2)

    return ab_img, l_img


def build_HDF5(jpeg_dir, size=256):
    """Gather the data in a single HDF5 file."""
    # Put train data in HDF5
    file_name = os.path.basename(jpeg_dir.rstrip("/"))
    hdf5_file = os.path.join(data_dir, "%s_data.h5" % file_name)
    with h5py.File(hdf5_file, "w") as hfw:
        for dset_type in ["train", "test", "valid"]:
            print('Working on {}'.format(dset_type))
            img_list = Path(jpeg_dir).glob('%s/*.jpg' % dset_type)
            img_list = np.array(list(map(str, img_list)))

            data_ab = hfw.create_dataset("%s_data_ab" % dset_type,
                                         (0, 2, size, size),
                                         maxshape=(None, 2, size, size),
                                         dtype=np.uint8)

            data_l = hfw.create_dataset("%s_data_l" % dset_type,
                                        (0, 1, size, size),
                                        maxshape=(None, 1, size, size),
                                        dtype=np.uint8)

            num_files = len(img_list)
            chunk_size = 100
            num_chunks = num_files // chunk_size
            print("Found {} images that will be processed in {} chunks".format(num_files, num_chunks))

            arr_chunks = np.array_split(np.arange(num_files), num_chunks)

            for chunk_idx in tqdm(arr_chunks):
                list_img_path = img_list[chunk_idx].tolist()
                output = parmap.map(format_image, list_img_path, size, parallel=False)

                arr_img_ab = np.concatenate([o[0] for o in output], axis=0)
                arr_img_l = np.concatenate([o[1] for o in output], axis=0)

                # Resize HDF5 dataset
                data_ab.resize(data_ab.shape[0] + arr_img_ab.shape[0], axis=0)
                data_l.resize(data_l.shape[0] + arr_img_l.shape[0], axis=0)

                data_ab[-arr_img_ab.shape[0]:] = arr_img_ab.astype(np.uint8)
                data_l[-arr_img_l.shape[0]:] = arr_img_l.astype(np.uint8)


def check_HDF5(jpeg_dir):
    """Plot images with landmarks to check the processing"""
    # Get hdf5 file
    file_name = os.path.basename(jpeg_dir.rstrip("/"))
    hdf5_file = os.path.join(data_dir, "%s_data.h5" % file_name)

    with h5py.File(hdf5_file, "r") as hf:
        data_ab = hf["train_data_ab"]
        data_l = hf["train_data_l"]
        for i in range(data_ab.shape[0]):
            plt.figure()
            img_ab = data_ab[i, :, :, :].transpose(1, 2, 0)
            img_l = data_l[i, :, :, :].transpose(1, 2, 0)

            img = np.concatenate((img_l, img_ab), axis=-1)
            img2 = np.zeros_like(img)
            img2[:, :, 0] = img_l[:, :, 0]

            img_both = np.concatenate((img, img2), axis=1)
            plt.imshow(cv2.cvtColor(img_both, cv2.COLOR_Lab2RGB))
            plt.show()
            plt.clf()
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build dataset')
    parser.add_argument('jpeg_dir', type=str, help='path to jpeg images')
    parser.add_argument('--img_size', default=256, type=int,
                        help='Desired Width == Height')
    parser.add_argument('--do_plot', action="store_true",
                        help='Plot the images to make sure the data processing went OK')
    args = parser.parse_args()

    data_dir = "../../data/processed"

    # build_HDF5(args.jpeg_dir, size=args.img_size)

    if args.do_plot:
        check_HDF5(args.jpeg_dir)
