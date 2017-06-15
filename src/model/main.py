import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from general_utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(prog='pix2pix')

    # General options
    parser.add_argument('patch_size', type=int, nargs=2, action="store", help="Patch size for D")
    parser.add_argument('--generator', default='upsampling', choices=['upsampling', 'deconv'], help="Generator type")
    parser.add_argument('--dset', default="/input/baikal.h5", help="Path to dataset")
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--use_mbd', action="store_true", help="Whether to use minibatch discrimination")
    subparsers = parser.add_subparsers(dest='mode')

    # Train options
    parser_train = subparsers.add_parser('train', help='Train GAN model')
    parser_train.add_argument('--n_batch_per_epoch', default=500, type=int, help="Number of batches per epoch")
    parser_train.add_argument('--nb_epoch', default=10, type=int, help="Number of training epochs")
    parser_train.add_argument('--epoch', default=1, type=int, help="Epoch at which weights were saved for evaluation")
    parser_train.add_argument('--label_smoothing', action="store_true",
                              help="Whether to smooth the positive labels when training D")
    parser_train.add_argument('--label_flipping', default=0, type=float,
                              help="Probability (0 to 1.) to flip the labels when training D")
    # Wasserstein options
    parser_train.add_argument('--use_wass', action='store_true', help='Use Wasserstein GAN')
    parser_train.add_argument('--disc_steps', type=int, default=1,
                              help='Specify the number of discriminator training steps')
    parser_train.add_argument('--alpha', type=float, default=1E-3, help='Learning rate')
    parser_train.add_argument('--clip', type=float, default=1e-2, help='Clipping parameter')

    # Apply options
    parser_apply = subparsers.add_parser('apply', help='Apply pretrained model')
    parser_apply.add_argument('--nb_batch', default=1, type=int, help="Number of batches")
    parser_apply.add_argument('--weights', required=True, help='Path to generator model weights')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Setup environment (logging directory etc)
    setup_logging()

    if args.mode == 'train':
        from train import train
        train(args.patch_size, args.generator, args.dset, args.batch_size, args.n_batch_per_epoch,
              args.nb_epoch, args.epoch, args.use_mbd, args.label_smoothing, args.label_flipping,
              args.use_wass, args.disc_steps, args.alpha, args.clip)
    elif args.mode == 'apply':
        from apply_net import apply_net
        apply_net(args.patch_size, args.generator, args.dset, args.batch_size,
                  args.use_mbd, args.nb_batch, args.weights)
