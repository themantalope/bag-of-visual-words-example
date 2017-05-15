from . import patchifier
import numpy as np

class ImagePatchifier(object):

    def __init__(self):
        self.patchifier = patchifier.Patchifier()
