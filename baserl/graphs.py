import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
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
                   title="policy",
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
    plt.title(title)
    plt.show()


def wireframe_value_function(value_function,
                             print_format=None,
                             mapping_key_func = lambda k: k,
                             make_default_key_func = lambda k: k,
                             view_elev_angle=30,
                             view_azim_angle=210,
                             title="Value Function"):
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
    x_axis = np.arange(minx, maxx + 1)
    y_axis = np.arange(miny, maxy + 1)
    x_axis, y_axis = np.meshgrid(x_axis, y_axis)
    z_axis = np.zeros((maxx + 1 -minx, maxy + 1 - miny))
    for x in range(minx, maxx + 1):
        for y in range(miny, maxy + 1):
            val = 0
            key = make_default_key_func((x, y)) 
            if key in value_function:
                val = value_function[key]
            z_axis[x - minx,y - miny] = val
            if print_format is not None:
                sys.stdout.write(print_format % val)
                sys.stdout.write(" ")
        if print_format is not None:
            sys.stdout.write("\n")            
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_wireframe(X=x_axis, Y=y_axis, Z=z_axis.T, rstride=1, cstride=1)
    ax.view_init(view_elev_angle, view_azim_angle)
    plt.title(title)
    plt.show()

def heatmap_q_value(q, missing_value, print_format=None,
                           mapping_key_func=lambda k: k,
                           inv_mapping_key_func=lambda k: k,
                           mapping_value_func=lambda k: k):
    xs = set()
    ys = set()
    for key in q:
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
            max_a = None
            max_v = None
            if inv_mapping_key_func((x, y)) not in q:
                max_v = missing_value
            else:
                for action, v in q[inv_mapping_key_func((x,y))].items():
                    v = mapping_value_func(v)
                    if max_v is None or max_v < v:
                        max_v = v
                        max_a = action
            a[x-minx,y-miny] = max_v
            if print_format is not None:
                sys.stdout.write(print_format.format(max_v))
                sys.stdout.write(" ")
        if print_format is not None:
            sys.stdout.write("\n")
    plt.imshow(a, cmap=plt.cm.rainbow, interpolation='nearest')
    plt.colorbar()
    plt.title('Q-values')
    plt.show()
    
def heatmap_q_value_delta(q, missing_value, print_format=None,
                           mapping_key_func=lambda k: k,
                           inv_mapping_key_func=lambda k: k,
                           mapping_value_func=lambda k: k):
    """
    Computes the delta between values for top two actions
    """
    xs = set()
    ys = set()
    for key in q:
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
            sorted_actions = sorted(q[inv_mapping_key_func((x,y))].items(), key=lambda x:x[1], reverse=True)
            first = mapping_value_func(sorted_actions[0][1])
            second = 0
            if len(sorted_actions) > 1:
                second = mapping_value_func(sorted_actions[1][1])
            val = (first - second) * 1.0 / (first+second)
            a[x-minx,y-miny] = val
            print(val, first, second)
            if print_format is not None:
                sys.stdout.write(print_format.format(max_v))
                sys.stdout.write(" ")
        if print_format is not None:
            sys.stdout.write("\n")
    plt.imshow(a, cmap=plt.cm.rainbow, interpolation='nearest')
    plt.colorbar()
    plt.title('Q-values')
    plt.show()

def heatmap_q_value_for_action(q, action, missing_value, print_format=None,
                           mapping_key_func=lambda k: k,
                           inv_mapping_key_func=lambda k: k,
                           mapping_value_func=lambda k: k,
                           title="Q-values"):
    xs = set()
    ys = set()
    for key in q:
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
            val = missing_value
            if action in q[inv_mapping_key_func((x,y))]:
                val = mapping_value_func(q[inv_mapping_key_func((x,y))][action])
            a[x-minx,y-miny] = val
            if print_format is not None:
                sys.stdout.write(print_format.format(max_v))
                sys.stdout.write(" ")
        if print_format is not None:
            sys.stdout.write("\n")
    plt.imshow(a, cmap=plt.cm.rainbow, interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()
