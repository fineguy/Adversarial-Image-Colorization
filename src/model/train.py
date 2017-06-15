import os
import sys
import time
import numpy as np

from models import load_model, DCGAN

from keras.utils import generic_utils
from keras.optimizers import Adam, RMSprop

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from data_utils import load_data, get_nb_patch, gen_batch, get_disc_batch, plot_generated_batch


def train_disc_reg(train_batch_generator, generator_model, discriminator_model,
                   batch_counter, patch_size, label_smoothing, label_flipping):
    X_ab_batch, X_l_batch = next(train_batch_generator)

    # Create a batch to feed the discriminator model
    X_disc, y_disc = get_disc_batch(X_ab_batch, X_l_batch, generator_model, batch_counter, patch_size,
                                    label_smoothing, label_flipping)

    # Update the discriminator
    disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

    return disc_loss


def train_disc_wass(train_batch_generator, generator_model, discriminator_model,
                    patch_size, label_smoothing, label_flipping, disc_steps, clip):
    """This is not exactly how the original Wasserstein GAN is trained, but should work nonetheless."""
    disc_loss_gen, disc_loss_real = [None] * disc_steps, [None] * disc_steps

    for i in range(disc_steps):
        X_ab_batch, X_l_batch = next(train_batch_generator)

        # Create batches to feed the discriminator model
        X_disc_gen, y_disc_gen = get_disc_batch(X_ab_batch, X_l_batch, generator_model, 0, patch_size,
                                                label_smoothing, label_flipping)
        X_disc_real, y_disc_real = get_disc_batch(X_ab_batch, X_l_batch, generator_model, 0, patch_size,
                                                  label_smoothing, label_flipping)

        # Update the discriminator
        disc_loss_gen[i] = discriminator_model.train_on_batch(X_disc_gen, y_disc_gen)
        disc_loss_real[i] = discriminator_model.train_on_batch(X_disc_real, y_disc_real)

        # Clip discriminator weights
        for layer in discriminator_model.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -clip, clip) for w in weights]
            layer.set_weights(weights)

    return np.mean(disc_loss_gen + disc_loss_real)


def train(patch_size, generator, dset, batch_size, n_batch_per_epoch,
          nb_epoch, epoch, use_mbd, label_smoothing, label_flipping,
          use_wass, disc_steps, alpha, clip):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """
    epoch_size = n_batch_per_epoch * batch_size

    # Load and rescale data
    X_ab_train, X_l_train = load_data(dset, 'train')
    X_ab_valid, X_l_valid = load_data(dset, 'valid')
    img_dim = X_l_train.shape[-3:]

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = get_nb_patch(X_ab_train.shape[-3:], patch_size)

    try:
        print('Creating optimizers')
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        if use_wass:
            print('Using RMSprop')
            opt_discriminator = RMSprop(lr=alpha, rho=0.9, epsilon=1e-08, decay=0.0)
        else:
            print('Using Adam')
            opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        print('Loading generator model')
        generator_model = load_model("generator_unet_{}".format(generator), img_dim, nb_patch, use_mbd, batch_size)
        print('Loading discriminator model')
        discriminator_model = load_model("DCGAN_discriminator", img_dim_disc, nb_patch, use_mbd, batch_size)

        print('Loading DCGAN model')
        generator_model.compile(loss='mae', optimizer=opt_discriminator)
        discriminator_model.trainable = False

        DCGAN_model = DCGAN(generator_model, discriminator_model, img_dim, patch_size)

        loss = ['mae', 'binary_crossentropy']
        loss_weights = [1E2, 1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        discriminator_model.trainable = True
        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        train_batch_generator = gen_batch(X_ab_train, X_l_train, batch_size)
        valid_batch_generator = gen_batch(X_ab_valid, X_l_valid, batch_size)

        disc_loss_list = list()
        gen_loss_list = list()
        time_list = list()

        start = time.time()
        print("Start training")
        for e in range(nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            epoch_start = time.time()

            for batch_counter in range(n_batch_per_epoch):
                if use_wass:
                    disc_loss = train_disc_wass(train_batch_generator, generator_model, discriminator_model,
                                                patch_size, label_smoothing, label_flipping, disc_steps, clip)
                else:
                    disc_loss = train_disc_reg(train_batch_generator, generator_model, discriminator_model,
                                               batch_counter, patch_size, label_smoothing, label_flipping)

                # Create a batch to feed the generator model
                X_ab_batch, X_l_batch = next(train_batch_generator)
                y_gen = np.zeros((X_l_batch.shape[0], 2), dtype=np.uint8)
                y_gen[:, 1] = 1

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch(X_l_batch, [X_ab_batch, y_gen])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                progbar.add(batch_size, values=[("D logloss", disc_loss),
                                                ("G tot", gen_loss[0]),
                                                ("G L1", gen_loss[1]),
                                                ("G logloss", gen_loss[2])])
                disc_loss_list.append(disc_loss)
                gen_loss_list.append(gen_loss)
                time_list.append(time.time() - start)

                # Save images for visualization 2 times per epoch
                if batch_counter % (n_batch_per_epoch // 2) == 0:
                    # Get new images from validation
                    plot_generated_batch(X_ab_batch, X_l_batch, generator_model, batch_size,
                                         "training_{}_{}".format(e, batch_counter))
                    X_ab_batch, X_l_batch = next(valid_batch_generator)
                    plot_generated_batch(X_ab_batch, X_l_batch, generator_model, batch_size,
                                         "validation_{}_{}".format(e, batch_counter))

                # Print batch training times only during the first epoch
                if e == 0 and (batch_counter + 1) % 100 == 0:
                    print('Batch {}/{}, Time: {:.2f}'.format(
                        batch_counter + 1, n_batch_per_epoch, time.time() - epoch_start))

            print("")
            print('Epoch {}/{}, Time: {:.2f}'.format(e + 1, nb_epoch, time.time() - epoch_start))

            # save weights each epoch
            if e % epoch == 0:
                gen_weights_path = os.path.join('/output/models/gen_weights_epoch_{}.h5'.format(e))
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join('/output/models/disc_weights_epoch_{}.h5'.format(e))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join('/output/models/DCGAN_weights_epoch_{}.h5'.format(e))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

        np.save('/output/models/disc_loss.npy', disc_loss_list)
        np.save('/output/models/gen_loss.npy', gen_loss_list)
        np.save('/output/models/times.npy', time_list)

    except KeyboardInterrupt:
        pass
