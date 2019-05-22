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

#$$ images --> video
import cv2
from PIL import Image

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    #url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    #with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    modelname = '../network-snapshot-005726.pkl'
    with open(modelname, "rb") as f:
        _G, _D, Gs = pickle.load(f)
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    os.makedirs(config.result_dir, exist_ok=True)

    # $$ images --> video
    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 30
    width = 704
    height = 1024
    videowriter = cv2.VideoWriter(os.path.join(config.result_dir, 'video_'+modelname.split('-')[-1]+'.avi'),
                                  fourcc, fps, (width, height))

    idx = 0
    rnd = np.random.RandomState(5)
    from_latents = rnd.randn(1, Gs.input_shape[1])
    for i in range(100): # target number
        to_latents = rnd.randn(1, Gs.input_shape[1])

        print('i = {}'.format(i))

        change = 5*fps # 10 is not enough, using 2*fps = 2 * 24~30
        for c in range(change):
            # Pick latent vector.
            #rnd = np.random.RandomState(5)
            this_latents = (1.-float(c)/change) * from_latents + (float(c)/change) * to_latents
            latents = this_latents

            # Generate image.
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

            # Save image.
            #os.makedirs(config.result_dir, exist_ok=True)
            idx += 1
            png_filename = os.path.join(config.result_dir, 'example_{}.png'.format(idx))
            img = PIL.Image.fromarray(images[0], 'RGB')
            img.save(png_filename)

            # $$ images --> video
            imgz = img.resize((int(width), int(height)), Image.ANTIALIAS)
            frame = cv2.cvtColor(np.asarray(imgz), cv2.COLOR_RGB2BGR)
            videowriter.write(frame)

        from_latents = to_latents

    videowriter.release()

if __name__ == "__main__":
    main()
