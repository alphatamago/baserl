import sys

import matplotlib.pyplot as plt
import numpy as np

def heatmap_value_function(value_function,
                           print_format=None,
                           mapping_key_func = lambda k: k,
                           make_default_key_func = lambda k: k):
    """
    mapping_key_func extracts the (x,y) coordinates for 2-D state space (in case
    the state is more complex)
    make_default_key_func mapx (x,y) to a default state value (in case the state
    is more complex)
    """
    xs = set()
    ys = set()
    for key in value_function:
        (x, y) = mapping_key_func(key)
        xs.add(x)
        ys.add(y)
    minx = min(xs)
    maxx = max(xs)
    miny = min(ys)
    maxy = max(ys)
    a = np.zeros((maxx + 1 -minx, maxy + 1 - miny))
    for x in range(minx, maxx + 1):
        for y in range(miny, maxy + 1):
            val = 0
            key = make_default_key_func((x, y)) 
            if key in value_function:
                val = value_function[key]
            a[x - minx,y - miny] = val
            if print_format is not None:
                sys.stdout.write(print_format % val)
                sys.stdout.write(" ")
        if print_format is not None:
            sys.stdout.write("\n")
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('value function')
    plt.show()
    
def heatmap_policy(policy,
                   print_format=None,
                   default_action_if_missing=0,
                   mapping_key_func=lambda k: k,
                   inv_mapping_key_func=lambda k: k,
                   mapping_value_func=lambda k: k):
    xs = set()
    ys = set()
    for key in policy:
        (x, y) = mapping_key_func(key)
        xs.add(x)
        ys.add(y)
    minx = min(xs)
    maxx = max(xs)
    miny = min(ys)
    maxy = max(ys)
    a = np.zeros((maxx + 1 -minx, maxy + 1 - miny))
    for x in range(minx, maxx + 1):
        for y in range(miny, maxy + 1):
            if inv_mapping_key_func((x,y)) not in policy:
                max_a = default_action_if_missing
            else:
                max_a = None
                max_v = None
                for action, v in policy[inv_mapping_key_func((x,y))].items():
                    v = mapping_value_func(v)
                    if max_v is None or max_v < v:
                        max_v = v
                        max_a = action
            a[x-minx,y-miny] = max_a
            if print_format is not None:
                sys.stdout.write(print_format % max_a)
                sys.stdout.write(" ")
        if print_format is not None:
            sys.stdout.write("\n")
    plt.imshow(a, cmap=plt.cm.rainbow, interpolation='nearest')
    plt.colorbar()
    plt.title('policy')
    plt.show()
