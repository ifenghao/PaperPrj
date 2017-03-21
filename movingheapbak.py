# coding:utf-8
import heapq
from copy import copy
import numpy as np


def push_active_2d(hp, x, x_probs, y, y_probs, pos, x_active, y_active):
    z = x[pos[0]] + y[pos[1]]
    prob = x_probs[pos[0]] * y_probs[pos[1]]
    heapq.heappush(hp, (z, prob, pos))
    x_active[pos[0]] = 1
    y_active[pos[1]] = 1


def pop_update_2d(hp, x, x_probs, y, y_probs, x_active, y_active):
    z, prob, pos = heapq.heappop(hp)
    x_active[pos[0]] = 0
    y_active[pos[1]] = 0
    new_pos = (pos[0] + 1, pos[1])
    if new_pos[0] < len(x) and x_active[new_pos[0]] == 0 and y_active[new_pos[1]] == 0:
        push_active_2d(hp, x, x_probs, y, y_probs, new_pos, x_active, y_active)
    new_pos = (pos[0], pos[1] + 1)
    if new_pos[1] < len(y) and x_active[new_pos[0]] == 0 and y_active[new_pos[1]] == 0:
        push_active_2d(hp, x, x_probs, y, y_probs, new_pos, x_active, y_active)
    return z, prob


def moving_heap_2d(x, x_probs, y, y_probs):  # x,y都是升序排列
    hp = []
    z = []
    z_probs = []
    x_active = np.zeros(len(x))
    y_active = np.zeros(len(y))
    push_active_2d(hp, x, x_probs, y, y_probs, (0, 0), x_active, y_active)
    while len(hp) != 0:
        new_z, prob = pop_update_2d(hp, x, x_probs, y, y_probs, x_active, y_active)
        if len(z) == 0:
            z.append(new_z)
            z_probs.append(prob)
            pre_z = new_z
        else:
            if np.allclose(pre_z, new_z):  # 相同的元素概率合并
                z_probs[-1] += prob
            else:
                z.append(new_z)
                z_probs.append(prob)
                pre_z = new_z
    return z, z_probs


def depth_first(x_lists, x_probs_lists):
    ndim = len(x_lists)
    global positive_prob
    positive_prob = 0

    def dfs(x_sum, x_prob, idim):
        global positive_prob
        if idim == ndim:
            if x_sum > 0:
                positive_prob += x_prob
        else:
            for x, x_p in zip(x_lists[idim], x_probs_lists[idim]):
                new_sum = x_sum + x
                new_prob = x_prob * x_p
                dfs(new_sum, new_prob, idim + 1)

    dfs(0., 1., 0)
    return positive_prob


def combine(ndim_list):  # 输入每列维度的列表
    depth = len(ndim_list)
    idx_list = []

    def dfs(lst):
        idim = len(lst)
        if idim == depth:
            idx_list.append(lst)
        else:
            for i in xrange(ndim_list[idim]):
                new_lst = copy(lst)
                new_lst.append(i)
                dfs(new_lst)

    dfs([])
    return idx_list


def bit_add(bit, result, ndim_list):
    if bit < 0:
        return
    result[bit] += 1
    if result[bit] >= ndim_list[bit]:
        result[bit] = 0
        bit_add(bit - 1, result, ndim_list)


def combine_gen(ndim_list):  # 输入每列维度的列表
    nbit = len(ndim_list)
    result = [0, ] * nbit
    while True:
        yield result
        bit_add(nbit - 1, result, ndim_list)
        if sum(result) == 0: break


def brute_force(x_lists, x_probs_lists):
    ndim_list = map(lambda x: len(x), x_lists)
    idx_list_gen = combine_gen(ndim_list)
    positive_prob = 0
    for idx_list in idx_list_gen:
        z = 0
        for num, idx in enumerate(idx_list):
            z += x_lists[num][idx]
        if z > 0:
            z_prob = 1
            for num, idx in enumerate(idx_list):
                z_prob *= x_probs_lists[num][idx]
            positive_prob += z_prob
    return positive_prob


def push_active_md(hp, x_list, pos, x_active):
    z = 0.
    prob = 1.
    for i, x in zip(pos, x_list):
        z += x[0][i]
        prob *= x[1][i]
    heapq.heappush(hp, (z, prob, pos))
    for i, active in zip(pos, x_active):
        active[i] = 1


def isinactive(pos, x_active):
    for i, active in zip(pos, x_active):
        if active[i] == 1:
            return False
    return True


def isinbound(pos, x_list):
    for i in xrange(len(pos)):
        if pos[i] >= len(x_list[i][0]):
            return False
    return True


def gen_pos(ndim):
    def dfs(lst):
        if len(lst) == ndim:
            yield lst
        else:
            for i in xrange(2):
                new_lst = copy(lst)
                new_lst.append(i)
                dfs(new_lst)

    dfs([])


def pop_update_md(hp, x_list, x_active):
    z, prob, pos = heapq.heappop(hp)
    for i, active in zip(pos, x_active):
        active[i] = 0
    for i, new_pos in enumerate(gen_pos(len(x_list))):
        if i == 0 or i == 2 ** len(x_list) - 1:  # 第一个和最后一个删除
            continue
        if isinbound(new_pos, x_list) and isinactive(new_pos, x_active):
            push_active_md(hp, x_list, new_pos, x_active)
    return z, prob


def moving_heap_md(x_list):  # 升序排列
    hp = []
    z = []
    z_probs = []
    ndim = len(x_list)
    x_active = []
    for i in xrange(ndim):
        x_active.append(np.zeros(len(x_list[i][0])))
    push_active_md(hp, x_list, (0,) * ndim, x_active)
    while len(hp) != 0:
        new_z, prob = pop_update_md(hp, x_list, x_active)
        if len(z) == 0:
            z.append(new_z)
            z_probs.append(prob)
            pre_z = new_z
        else:
            if np.allclose(pre_z, new_z):  # 相同的元素概率合并
                z_probs[-1] += prob
            else:
                z.append(new_z)
                z_probs.append(prob)
                pre_z = new_z
    return z, z_probs
