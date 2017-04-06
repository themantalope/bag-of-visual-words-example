import numpy as np
import math

def _check_overlap(overlap):
    return 0.0 < overlap < 1.0

def _check_scaling_factor(scaling_factor):
    return scaling_factor > 1.0

def _patch_generator(im_shape,
                     patch_shape,
                     x_overlap,
                     y_overlap):

    if not _check_overlap(x_overlap) or not _check_overlap(y_overlap):
        raise TypeError("'x_overlap' and 'y_overlap' must be in (0.0, 1.0)")

    x_step = int(math.floor(patch_shape[0] * x_overlap))
    y_step = int(math.floor(patch_shape[1] * y_overlap))

    im_width = im_shape[1]
    im_height = im_shape[0]


    for i in range(0, im_width-x_step, x_step):
        for j in range(0, im_height-y_step, y_step):
            yield (i, i+x_step, j, j+y_step)

def _multiscale_patch_generator(im_shape,
                                largest_patch_size,
                                smallest_patch_size,
                                scaling_factor,
                                x_overlap,
                                y_overlap):
    if not _check_scaling_factor(scaling_factor):
        raise ValueError("'scaling_factor' must be > 1.0")

    scaled_patches = [smallest_patch_size]
    max_patch_x = largest_patch_size[1]
    max_patch_y = largest_patch_size[0]
    while True:
        last_patch = scaled_patches[-1]
        last_patch_x = last_patch[1]
        last_patch_y = last_patch[0]
        next_patch_x = int(math.floor(last_patch_x ** scaling_factor))
        next_patch_y = int(math.floor(last_patch_y ** scaling_factor))

        if (next_patch_x > max_patch_x) or (next_patch_y > max_patch_y):
            break
        else:
            scaled_patches.append((next_patch_y, next_patch_x))

    scaled_patches.sort(key= lambda x: x[0])
    for sp in scaled_patches:
        yield _patch_generator(im_shape, sp, x_overlap, y_overlap)


def make_patch_list(im_shape,
                    patch_shape,
                    x_overlap,
                    y_overlap):
    patchifer = Patchifier()
    plist = patchifer.get_patch_list(im_shape, patch_shape, x_overlap, y_overlap)
    return plist

def make_patch_generator(im_shape,
                         patch_shape,
                         x_overlap,
                         y_overlap):
    patchifer = Patchifier()
    pgen = patchifer.get_patch_generator(im_shape, patch_shape, x_overlap, y_overlap,return_generator=True)
    return pgen


def make_multiscale_patch_generator(im_shape,
                                    largest_patch_size,
                                    smallest_patch_size,
                                    scaling_factor,
                                    x_overlap,
                                    y_overlap):
    patchifer = Patchifier()
    ms_pgen = patchifer.get_multiscale_patch_generator(im_shape,
                                                       largest_patch_size,
                                                       smallest_patch_size,
                                                       scaling_factor,
                                                       x_overlap,
                                                       y_overlap,
                                                       return_generator=True)
    return ms_pgen

def make_multiscale_patch_array(im_shape,
                                largest_patch_size,
                                smallest_patch_size,
                                scaling_factor,
                                x_overlap,
                                y_overlap):
    patchifer = Patchifier()
    ms_parray = patchifer.get_multiscale_patch_array(im_shape,
                                                     largest_patch_size,
                                                     smallest_patch_size,
                                                     scaling_factor,
                                                     x_overlap,
                                                     y_overlap)
    return ms_parray



class Patchifier(object):

    def __init__(self):
        self.patch_generator = None

    def get_patch_generator(self,
                            im_shape,
                            patch_shape,
                            x_overlap,
                            y_overlap,
                            return_generator=False):



        self.patch_generator = _patch_generator(im_shape, patch_shape, x_overlap, y_overlap)
        if return_generator:
            return self.patch_generator

    def get_patch_list(self,
                       im_shape,
                       patch_shape,
                       x_overlap,
                       y_overlap):
        patch_list = []
        self.get_patch_generator(im_shape, patch_shape, x_overlap, y_overlap)
        for patch in self.patch_generator:
            patch_list.append(patch)

        return patch_list

    def get_next_patch(self,
                       end_return_value=None):
        if self.patch_generator:
            next_val = next(self.patch_generator, end_return_value)
            if next_val == end_return_value:
                self.patch_generator = None

        return next_val

    def get_multiscale_patch_generator(self,
                                       im_shape,
                                       largest_patch_size,
                                       smallest_patch_size,
                                       scaling_factor,
                                       x_overlap,
                                       y_overlap,
                                       return_generator=False):
        """
        scales the current patch size in each dimension by the exp of the
        scaling factor, until it is larger than the largest allowed patch
        size. will stop once the patch size is larger than one of the
        dimensions specified in the 'largest_patch_size' variable
        """

        self.multiscale_patch_generator = _multiscale_patch_generator(im_shape,
                                                                      largest_patch_size,
                                                                      smallest_patch_size,
                                                                      scaling_factor,
                                                                      x_overlap,
                                                                      y_overlap)
        if return_generator:
            return self.multiscale_patch_generator

    def get_next_multiscale_patch_set(self,
                                      end_return_value=None):
        if self.multiscale_patch_generator:
            next_gen = next(self.multiscale_patch_generator, end_return_value)
            if next_gen == end_return_value:
                self.multiscale_patch_generator = None
                return end_return_value
            else:
                patches = [p for p in next_gen]
                return patches


    def get_multiscale_patch_array(self,
                                  im_shape,
                                  largest_patch_size,
                                  smallest_patch_size,
                                  scaling_factor,
                                  x_overlap,
                                  y_overlap):

        patch_array = []
        self.get_multiscale_patch_generator(im_shape,
                                            largest_patch_size,
                                            smallest_patch_size,
                                            scaling_factor,
                                            x_overlap,
                                            y_overlap)

        for gen in self.multiscale_patch_generator:
            cur_patches = [p for p in gen]
            patch_array.append(cur_patches)

        return patch_array
