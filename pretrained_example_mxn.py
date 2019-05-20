# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    #url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    #with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    with open('../network-snapshot-004085.pkl', "rb") as f:
        _G, _D, Gs = pickle.load(f)
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    idx = 0
    rnd = np.random.RandomState(5)
    from_latents = rnd.randn(1, Gs.input_shape[1])
    for i in range(30): # target number
        to_latents = rnd.randn(1, Gs.input_shape[1])

        change = 10
        for c in range(change):
            # Pick latent vector.
            #rnd = np.random.RandomState(5)
            this_latents = (1.-float(c)/change) * from_latents + (float(c)/change) * to_latents
            latents = this_latents

            # Generate image.
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

            # Save image.
            os.makedirs(config.result_dir, exist_ok=True)
            idx += 1
            png_filename = os.path.join(config.result_dir, 'example_{}.png'.format(idx))
            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        from_latents = to_latents

if __name__ == "__main__":
    main()
