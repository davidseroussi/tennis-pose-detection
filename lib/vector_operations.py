import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180/np.pi

def get_intersection_3d(point1, vec1, point2, vec2):
    
    """https://math.stackexchange.com/questions/270767/find-intersection-of-two-3d-lines/271366
    
    point1: 3D point on line 1
    vec1: 3D direction vector of line 1
    
    point2: 3D point on line 2
    vec2: 3D direction vector of line 2
    """
    
    C = point1
    D = point2

    e = vec1
    f = vec2
    g = D - C
    
    f_cross_g = np.cross(f, g)
    f_cross_e = np.cross(f, e)
    
    h = np.linalg.norm(f_cross_g)
    k = np.linalg.norm(f_cross_e)
        
    # if h == 0 or k == 0:
    #     return None
    
    f_g_orientation = f_cross_g / np.abs(f_cross_g)
    f_e_orientation = f_cross_e / np.abs(f_cross_e)
    
    sign = 1 if np.array_equal(f_g_orientation, f_e_orientation) else -1
        
    return C + sign * (h/k) * e