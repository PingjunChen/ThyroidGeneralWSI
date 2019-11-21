# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io
import openslide


if __name__ == "__main__":
    np.random.seed(1234)

    slide_level=4
    slide_dir = "../data/TestSlides/Malignant"
    slide_name = "1242672-4.tiff"

    thumb_dir = "../data/TestSlides/Thumbnail"
    if not os.path.exists(thumb_dir):
        os.makedirs(thumb_dir)
    slide_path = os.path.join(slide_dir, slide_name)
    slide_head = openslide.OpenSlide(slide_path)
    slide_img = slide_head.read_region(location=(0, 0), level=slide_level,
                                       size=slide_head.level_dimensions[slide_level])
    slide_img = np.asarray(slide_img)[:, :, :3]
    thumb_path = os.path.join(thumb_dir, os.path.splitext(slide_name)[0]+".png")
    io.imsave(thumb_path, slide_img)
