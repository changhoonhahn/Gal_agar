import os 
import numpy as np 
from numpy import Inf



def code_dir(): 
    return os.path.dirname(os.path.realpath(__file__)).split('util')[0]


def elements(vals, lim=[-Inf, Inf], vis=None, vis_2=None, get_indices=False, dtype=np.int32):
    '''
    Get the indices of the input values that are within the input limit, that also are in input vis
    index array (if defined).
    Either of limits can have same range as vals.

    Import array, range to keep, prior indices of vals array to keep,
    other array to sub-sample in same way, whether to return selection indices of input vis array.
    '''
    if not isinstance(vals, np.ndarray):
        vals = np.array(vals)
    # check if input array
    if vis is None:
        vis = np.arange(vals.size, dtype=dtype)
    else:
        vals = vals[vis]
    vis_keep = vis
    # check if limit is just one value
    if np.isscalar(lim):
        keeps = (vals == lim)
    else:
        # sanity check - can delete this eventually
        if isinstance(lim[0], int) and isinstance(lim[1], int):
            if lim[0] == lim[1]:
                raise ValueError('input limit = %s, has same value' % lim)
            if lim[0] != lim[1] and 'int' in vals.dtype.name:
                print '! elements will not keep objects at lim[1] = %d' % lim[1]
        if not np.isscalar(lim[0]) or lim[0] > -Inf:
            keeps = (vals >= lim[0])
        else:
            keeps = None
        if not np.isscalar(lim[1]) or lim[1] < Inf:
            if keeps is None:
                keeps = (vals < lim[1])
            else:
                keeps *= (vals < lim[1])
        elif keeps is None:
            keeps = np.arange(vals.size, dtype=dtype)
    if get_indices:
        if vis_2 is not None:
            return vis_keep[keeps], vis_2[keeps], np.arange(vis.size, dtype=dtype)[keeps]
        else:
            return vis_keep[keeps], np.arange(vis.size, dtype=dtype)[keeps]
    else:
        if vis_2 is not None:
            return vis_keep[keeps], vis_2[keeps]
        else:
            return vis_keep[keeps]

