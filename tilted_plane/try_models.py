# -*- coding: utf-8 -*-
"""
Try running the Perez models
"""

import numpy as np

from tilted_plane.perez_1990 import perez_1990
from tilted_plane.perez_1993 import perez_1993

if __name__ == "__main__":
    bh = np.array([600, 700])
    dh = 100
    sza = np.radians(30)
    saa = np.pi
    pza = np.radians(45)
    paa = np.pi
    albedo = 0.1
    doy = 1

    gt, bt, dt, rt = perez_1990(bh, dh, sza, saa, pza, paa, albedo, doy)
    gt93, bt93, drt93 = perez_1993(bh, dh, sza, saa, pza, paa, albedo, doy)

    for output in (gt, bt, dt, rt, gt93, bt93, drt93):
        print(output.shape)
    print("Perez 1990 global:", gt)
    print("Perez 1993 global:", gt93)
    print("Perez 1990 diffuse + reflected:", dt + rt)
    print("Perez 1993 diffuse + reflected:", drt93)

    print("Done")
