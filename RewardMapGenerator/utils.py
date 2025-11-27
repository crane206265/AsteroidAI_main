import numpy as np
from numpy import linalg as LA


########## Coord. Transformation ##########

def sph2cart(sph_coord):
    """
    spherical coord -> cartesian coord
    input : sph_coord = (r, phi, theta)
    output : (x, y, z)
    """
    r = sph_coord[0]
    phi = sph_coord[1]
    theta = sph_coord[2]
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.array([x, y, z])

def cart2sph(cart_coord):
    """
    cartesian coord -> spherical coord
    input : cart_coord = (x, y, z)
    output : (r, phi, theta)
    """
    x = cart_coord[0]
    y = cart_coord[1]
    z = cart_coord[2]
    r = LA.norm(np.array([x, y, z]))
    phi = np.arctan2(y, x)
    theta = np.arccos(z/r)
    return np.array([r, phi, theta])


########### -- ##########

def rotArr(angle, axis):
    # for rotational matrix
    if axis == "x" or axis == 0:
        arr = np.array([[1, 0            , 0             ],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle) ]])
    elif axis == "y" or axis == 1:
        arr = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0            , 1, 0             ],
                        [-np.sin(angle), 0, np.cos(angle) ]])
    elif axis == "z" or axis == 2:
        arr = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle) , 0],
                        [0            , 0             , 1]])
    else:
        raise ValueError("Unappropriate axis")
    return arr

def ReLU(x):
    return x * (x > 0)