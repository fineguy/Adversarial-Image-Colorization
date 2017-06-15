import os
import sys
import time

from models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from data_utils import load_data, get_nb_patch, gen_batch, plot_generated_batch


def apply_net(patch_size, generator, dset, batch_size, use_mbd, nb_batch, weights):
    # Load and rescale data
    X_ab_test, X_l_test = load_data(dset, 'test')
    img_dim = X_l_test.shape[-3:]

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = get_nb_patch(X_ab_test.shape[-3:], patch_size)

    # Load generator model
    generator_model = load_model("generator_unet_{}".format(generator), img_dim, nb_patch, use_mbd, batch_size)
    generator_model.load_weights(weights)
    batch_generator = gen_batch(X_ab_test, X_l_test, batch_size)
    start = time.time()

    for batch_counter in range(nb_batch):
        X_ab_batch, X_l_batch = next(batch_generator)
        plot_generated_batch(X_ab_batch, X_l_batch, generator_model, batch_size, "batch_{}".format(batch_counter))
        print('Batch #{}, Time: {:.2f}'.format(batch_counter + 1, time.time() - start))
