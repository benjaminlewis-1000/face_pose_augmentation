
import math
import numpy as np
from scipy.spatial import Delaunay
from collections import defaultdict
from typing import Dict, Tuple, Sequence, Optional, Union, List
import pickle
import time

def adjust_rotated_anchors_orig(all_vertex_src: np.ndarray, all_vertex_ref: np.ndarray, all_vertex_adjust: np.ndarray,
                           bg_tri: np.ndarray, anchor_flags: np.ndarray) -> np.ndarray:
    # Solve the equation Y = AX for x and y coordinates
    y_equ = []
    a_equ = []


    data = {"all_vertex_src": all_vertex_src,
        "all_vertex_ref": all_vertex_ref,
        "all_vertex_adjust": all_vertex_adjust,
        "bg_tri": bg_tri,
        "anchor_flags": anchor_flags}
    # for each outpoint
    adjust_ind = np.where(np.any([anchor_flags == 2, anchor_flags == 3], axis=0))[0]
    for pt in adjust_ind:
        # find the corresponding tri
        tmp_bin = np.any(bg_tri == pt, axis=0)

        # find connecting point
        temp = bg_tri[:, tmp_bin]
        connect = np.unique(temp)
        connect = connect[connect != pt]

        # the relationship of [pt, pt_con]
        for pt_con in connect:
            if anchor_flags[pt] == 2:
                # if base point is a src point, prefer src relation
                if anchor_flags[pt_con] == 1:
                    # if connect to a base point, module the positions
                    x_new = all_vertex_src[0, pt] - all_vertex_src[0, pt_con] + all_vertex_adjust[0, pt_con]
                    y_new = all_vertex_src[1, pt] - all_vertex_src[1, pt_con] + all_vertex_adjust[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)[0]

                    a_equ.append(np.zeros(shape=(2, 2 * len(adjust_ind))))
                    a_equ[-1][0, 2 * pt1] = 1
                    a_equ[-1][1, 2 * pt1 + 1] = 1
                    y_equ.extend([x_new, y_new])
                else:  # anchor_flags(pt_con) in [2, 3]
                    # src-src and src-ref relationships: based on src relationship
                    x_offset = all_vertex_src[0, pt] - all_vertex_src[0, pt_con]
                    y_offset = all_vertex_src[1, pt] - all_vertex_src[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)[0]
                    pt_con1 = np.where(adjust_ind == pt_con)[0]

                    a_equ.append(np.zeros(shape=(2, 2 * len(adjust_ind))))
                    a_equ[-1][0, 2 * pt1] = 1
                    a_equ[-1][0, 2 * pt_con1] = -1
                    a_equ[-1][1, 2 * pt1 + 1] = 1
                    a_equ[-1][1, 2 * pt_con1 + 1] = -1
                    y_equ.extend([x_offset, y_offset])
            else:  # anchor_flags(pt) == 3
                # if it is a ref point, prefer ref relation
                if anchor_flags[pt_con] == 1:
                    # if connect to a base point, module the positions
                    x_new = all_vertex_ref[0, pt] - all_vertex_ref[0, pt_con] + all_vertex_adjust[0, pt_con]
                    y_new = all_vertex_ref[1, pt] - all_vertex_ref[1, pt_con] + all_vertex_adjust[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)[0]

                    a_equ.append(np.zeros(shape=(2, 2 * len(adjust_ind))))
                    a_equ[-1][0, 2 * pt1] = 1
                    a_equ[-1][1, 2 * pt1 + 1] = 1
                    y_equ.extend([x_new, y_new])
                else:
                    # ref-ref relationships: based on ref relationship
                    x_offset = all_vertex_ref[0, pt] - all_vertex_ref[0, pt_con]
                    y_offset = all_vertex_ref[1, pt] - all_vertex_ref[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)[0]
                    pt_con1 = np.where(adjust_ind == pt_con)[0]

                    a_equ.append(np.zeros(shape=(2, 2 * len(adjust_ind))))
                    a_equ[-1][0, 2 * pt1] = 1
                    a_equ[-1][0, 2 * pt_con1] = -1
                    a_equ[-1][1, 2 * pt1 + 1] = 1
                    a_equ[-1][1, 2 * pt_con1 + 1] = -1
                    y_equ.extend([x_offset, y_offset])

    # get the new position
    x_equ = np.linalg.lstsq(np.vstack(a_equ), np.array(y_equ), rcond=None)[0]
    all_vertex_adjust[:2, adjust_ind] = x_equ.reshape((2, -1), order='F')
    all_vertex_adjust[2, adjust_ind] = all_vertex_ref[2, adjust_ind]

    return all_vertex_adjust


def adjust_rotated_anchors_mod(all_vertex_src: np.ndarray, all_vertex_ref: np.ndarray, all_vertex_adjust: np.ndarray,
                           bg_tri: np.ndarray, anchor_flags: np.ndarray) -> np.ndarray:
    # Solve the equation Y = AX for x and y coordinates
    y_equ = []
    a_equ = []


    data = {"all_vertex_src": all_vertex_src,
        "all_vertex_ref": all_vertex_ref,
        "all_vertex_adjust": all_vertex_adjust,
        "bg_tri": bg_tri,
        "anchor_flags": anchor_flags}
    # for each outpoint
    adjust_ind = np.where(np.any([anchor_flags == 2, anchor_flags == 3], axis=0))[0]

    print(np.any(bg_tri == adjust_ind, axis=0))
    print(np.any(bg_tri == adjust_ind[0], axis=0))
    for pt in adjust_ind:
        # find the corresponding tri
        tmp_bin = np.any(bg_tri == pt, axis=0)

        # find connecting point
        temp = bg_tri[:, tmp_bin]
        connect = np.unique(temp)
        connect = connect[connect != pt]

        # the relationship of [pt, pt_con]
        for pt_con in connect:
            if anchor_flags[pt] == 2:
                # if base point is a src point, prefer src relation
                if anchor_flags[pt_con] == 1:
                    # if connect to a base point, module the positions
                    x_new = all_vertex_src[0, pt] - all_vertex_src[0, pt_con] + all_vertex_adjust[0, pt_con]
                    y_new = all_vertex_src[1, pt] - all_vertex_src[1, pt_con] + all_vertex_adjust[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)[0]

                    a_equ.append(np.zeros(shape=(2, 2 * len(adjust_ind))))
                    a_equ[-1][0, 2 * pt1] = 1
                    a_equ[-1][1, 2 * pt1 + 1] = 1
                    y_equ.extend([x_new, y_new])
                else:  # anchor_flags(pt_con) in [2, 3]
                    # src-src and src-ref relationships: based on src relationship
                    x_offset = all_vertex_src[0, pt] - all_vertex_src[0, pt_con]
                    y_offset = all_vertex_src[1, pt] - all_vertex_src[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)[0]
                    pt_con1 = np.where(adjust_ind == pt_con)[0]

                    a_equ.append(np.zeros(shape=(2, 2 * len(adjust_ind))))
                    a_equ[-1][0, 2 * pt1] = 1
                    a_equ[-1][0, 2 * pt_con1] = -1
                    a_equ[-1][1, 2 * pt1 + 1] = 1
                    a_equ[-1][1, 2 * pt_con1 + 1] = -1
                    y_equ.extend([x_offset, y_offset])
            else:  # anchor_flags(pt) == 3
                # if it is a ref point, prefer ref relation
                if anchor_flags[pt_con] == 1:
                    # if connect to a base point, module the positions
                    x_new = all_vertex_ref[0, pt] - all_vertex_ref[0, pt_con] + all_vertex_adjust[0, pt_con]
                    y_new = all_vertex_ref[1, pt] - all_vertex_ref[1, pt_con] + all_vertex_adjust[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)[0]

                    a_equ.append(np.zeros(shape=(2, 2 * len(adjust_ind))))
                    a_equ[-1][0, 2 * pt1] = 1
                    a_equ[-1][1, 2 * pt1 + 1] = 1
                    y_equ.extend([x_new, y_new])
                else:
                    # ref-ref relationships: based on ref relationship
                    x_offset = all_vertex_ref[0, pt] - all_vertex_ref[0, pt_con]
                    y_offset = all_vertex_ref[1, pt] - all_vertex_ref[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)[0]
                    pt_con1 = np.where(adjust_ind == pt_con)[0]

                    a_equ.append(np.zeros(shape=(2, 2 * len(adjust_ind))))
                    a_equ[-1][0, 2 * pt1] = 1
                    a_equ[-1][0, 2 * pt_con1] = -1
                    a_equ[-1][1, 2 * pt1 + 1] = 1
                    a_equ[-1][1, 2 * pt_con1 + 1] = -1
                    y_equ.extend([x_offset, y_offset])

    # get the new position
    print(y_equ)
    lstsq_a = np.vstack(a_equ)
    lstsq_b = np.array(y_equ)
    x_equ = np.linalg.solve(lstsq_a.T.dot(lstsq_a), lstsq_a.T.dot(lstsq_b))
    all_vertex_adjust[:2, adjust_ind] = x_equ.reshape((2, -1), order='F')
    all_vertex_adjust[2, adjust_ind] = all_vertex_ref[2, adjust_ind]

    return all_vertex_adjust

if __name__ == '__main__':
    with open('inputs.pkl', 'rb') as fh:
        data = pickle.load(fh)
    adjust_rotated_anchors_orig(**data)

    n_rpt = 1

    s = time.time()
    for i in range(n_rpt):
        out = adjust_rotated_anchors_orig(**data)
    el = time.time() - s
    print(el / n_rpt)

    s = time.time()
    for i in range(n_rpt):
        out_mod = adjust_rotated_anchors_mod(**data)
    el = time.time() - s
    print(el / n_rpt)

    assert np.allclose(out, out_mod)

    all_vertex_src = data['all_vertex_src']
    all_vertex_ref = data['all_vertex_ref']
    all_vertex_adjust = data['all_vertex_adjust']
    bg_tri = data['bg_tri']
    anchor_flags = data['anchor_flags']

    adjust_ind = np.where(np.any([anchor_flags == 2, anchor_flags == 3], axis=0))[0]
    bg_tri_exp = np.expand_dims(bg_tri, axis=0)
    bg_tri_exp = np.repeat(bg_tri_exp, len(adjust_ind), axis=0)
    adj = np.expand_dims(adjust_ind, axis=1)
    adj = np.expand_dims(adj, axis=2)
    adj = np.repeat(adj, bg_tri.shape[0], axis=1)
    adj = np.repeat(adj, bg_tri.shape[1], axis=2)

    comp = bg_tri_exp == adj

    tmp_bin = np.any(comp,axis = 1)
    # find connecting point
    bg_shuffle = np.moveaxis(bg_tri_exp, 1, 0)

    temp = bg_shuffle[:, tmp_bin]
    # Used to separate back out. 
    per_line_counts = np.count_nonzero(tmp_bin, axis=1)
    max_dim = np.max(per_line_counts)
    num_pts = len(per_line_counts)

    per_array = np.zeros((num_pts, 3, max_dim))
    st = 0
    for idx, val in enumerate(per_line_counts):
        per_array[idx, :, :val] = temp[:, st:st + val]
        st = st + val

    per_array_reshape = np.reshape(per_array, (num_pts, -1))
    # NOT WORKING
    connect = np.unique(per_array_reshape, axis=0) 
    adj_cmp = np.expand_dims(adjust_ind, axis=1)
    adj_cmp = np.repeat(adj_cmp, max_dim * 3, axis=1)
    connect2 = connect[connect != adj_cmp]


    # connect = np.unique(temp)
    # connect = connect[connect != pt]

    pt = adjust_ind[0]
    tmp_bin_orig = np.any(bg_tri == pt, axis=0)
    # find connecting point
    temp_orig = bg_tri[:, tmp_bin_orig]
    connect_orig = np.unique(temp_orig)
    connect_orig = connect_orig[connect_orig != pt]
