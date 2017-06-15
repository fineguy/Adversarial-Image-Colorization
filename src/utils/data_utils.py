import cv2
import numpy as np
import h5py


def normalization(X):
    return X / 127.5 - 1


def inverse_normalization(X):
    return (X + 1) * 127.5


def get_nb_patch(img_dim, patch_size):
    assert img_dim[0] % patch_size[0] == 0, "patch_size does not divide height"
    assert img_dim[1] % patch_size[1] == 0, "patch_size does not divide width"
    nb_patch = (img_dim[0] // patch_size[0]) * (img_dim[1] // patch_size[1])
    img_dim_disc = (patch_size[0], patch_size[1], img_dim[-1])

    return nb_patch, img_dim_disc


def extract_patches(X, patch_size):
    return [X[:, i:i + patch_size[0], j:j + patch_size[1], :]
            for i in range(0, X.shape[1], patch_size[0])
            for j in range(0, X.shape[2], patch_size[1])]


def load_data(dset, mode):
    with h5py.File(dset, 'r') as hf:
        X_ab = hf["{}_data_ab".format(mode)][:].astype(np.float32)
        X_ab = normalization(X_ab)

        X_l = hf["{}_data_l".format(mode)][:].astype(np.float32)
        X_l = normalization(X_l)

        X_ab = X_ab.transpose(0, 2, 3, 1)
        X_l = X_l.transpose(0, 2, 3, 1)

        return X_ab, X_l


def gen_batch(X1, X2, batch_size):
    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        yield X1[idx], X2[idx]


def get_disc_batch(X_ab_batch, X_l_batch, generator_model, batch_counter, patch_size,
                   label_smoothing=False, label_flipping=0):
    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # Produce an output
        X_disc = generator_model.predict(X_l_batch)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        X_disc = X_ab_batch
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(X_disc, patch_size)

    return X_disc, y_disc


def plot_generated_batch(X_ab, X_l, generator_model, batch_size, suffix):
    # Generate images
    X_gen = generator_model.predict(X_l)

    # Limit to 8 pictures
    Xab = X_ab[:8]
    Xl = X_l[:8]
    Xg = X_gen[:8]

    Xg = np.concatenate((Xl, Xg), axis=-1)
    Xr = np.concatenate((Xl, Xab), axis=-1)
    Xl_ext = np.zeros_like(Xr)
    Xl_ext[:, :, :, 0] = Xl[:, :, :, 0]

    X = np.concatenate((Xl_ext, Xg, Xr), axis=2)
    X = np.concatenate(X, axis=0)
    X = inverse_normalization(X)

    cv2.imwrite("/output/figures/batch_{}.png".format(suffix), cv2.cvtColor(X.astype('uint8'), cv2.COLOR_Lab2BGR))
