import numpy as np
import torch



def rotoflip(image, rotoflip_id=None):
    '''
    Transforms an unlabeled image dataset into a labeled one by:
    * randomly rotating/flipping the image (there are 8 ways to do this)
    * returning tuples of `(image, rotoflip_id)`, where
      `rotoflip_id` is an integer from 0 through 7 identifying the rotation/flip combination 
    '''
    if rotoflip_id == None:
        rotoflip_id = np.random.randint(8)
    assert rotoflip_id >= 0
    assert rotoflip_id < 8
    if rotoflip_id < 4:
        return flip(image, rotoflip_id), rotoflip_id
    else:
        return flip(image.permute(0, 2, 1), rotoflip_id - 4), rotoflip_id


def flip(image, flip_id):
    assert flip_id >= 0
    assert flip_id < 4
    if flip_id == 0:
        return image
    if flip_id == 1:
        return image.flip(-1)
    if flip_id == 2:
        return image.flip(-2)
    if flip_id == 3:
        return image.flip(-2, -1)