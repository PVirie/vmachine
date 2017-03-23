import cv2
import numpy as np
import random
import math
import os
from Queue import PriorityQueue
from array2gif import write_gif


def search_path(minkowsky_map, start, end):

    def to_queue(p, prev, distp):
        if prev is None:
            return ((p[0] - end[0])**2 + (p[1] - end[1])**2, (p[0], p[1], 0))
        else:
            return ((p[0] - end[0])**2 + (p[1] - end[1])**2 + prev[2] + distp, (p[0], p[1], prev[2] + distp, prev))

    q = PriorityQueue()
    q.put(to_queue((start[0], start[1]), None, 0.0))

    def off_range(point):
        return p[1] < 0 or p[1] >= minkowsky_map.shape[0] or p[0] < 0 or p[0] >= minkowsky_map.shape[1]

    out = None
    visited = np.zeros((minkowsky_map.shape[0], minkowsky_map.shape[1]), dtype=np.float32)
    while not q.empty():
        p = q.get()[1]
        if p[0] == end[0] and p[1] == end[1]:
            out = p
            break
        if off_range(p) or visited[p[1], p[0]] > 0.5 or minkowsky_map[p[1], p[0]] > 0.5:
            continue
        visited[p[1], p[0]] = 1
        q.put(to_queue((p[0] + 0, p[1] - 1), p, 1))
        q.put(to_queue((p[0] - 1, p[1] + 0), p, 1))
        q.put(to_queue((p[0] + 1, p[1] + 0), p, 1))
        q.put(to_queue((p[0] + 0, p[1] + 1), p, 1))
        q.put(to_queue((p[0] - 1, p[1] - 1), p, math.sqrt(2)))
        q.put(to_queue((p[0] + 1, p[1] - 1), p, math.sqrt(2)))
        q.put(to_queue((p[0] - 1, p[1] + 1), p, math.sqrt(2)))
        q.put(to_queue((p[0] + 1, p[1] + 1), p, math.sqrt(2)))

    path = []
    if out is None:
        return path

    p = out
    while True:
        path.append((p[0], p[1]))
        if len(p) > 3:
            p = p[3]
        else:
            break
    path.reverse()
    return path


def draw_map(size, obj_size, map_complexity):
    canvas = np.zeros((size[0], size[1]), dtype=np.float32)
    minkowsky = np.zeros((size[0], size[1]), dtype=np.float32)

    for i in xrange(map_complexity):
        sx = random.randint(2, 15)
        sy = random.randint(2, 15)
        py = random.randint(5, size[0] - 5)
        px = random.randint(5, size[1] - 5)
        cv2.rectangle(canvas, (px - sx, py - sy), (px + sx, py + sy), (1.0, 1.0, 1.0), -1)
        cv2.rectangle(minkowsky, (px - sx - obj_size / 2, py - sy - obj_size / 2), (px + sx + obj_size / 2, py + sy + obj_size / 2), (1.0, 1.0, 1.0), -1)

    return canvas, minkowsky


def draw_path(space, obj_size, path, fixed_length=20):
    canvas = np.zeros((fixed_length, space.shape[0], space.shape[1]), dtype=np.float32)
    step = (len(path) - 1) * 1.0 / (fixed_length - 1)
    for i in xrange(0, fixed_length):
        canvas[i] = space
        p = path[int(i * step)]
        cv2.circle(canvas[i], (p[0], p[1]), obj_size / 2, (1.0, 1.0, 1.0), -1)
    return canvas


def plot_path(path):
    temp = np.copy(path)
    temp[:, 0, :] = 1
    temp[:, :, 0] = 1
    cv2.imshow("a", np.reshape(np.transpose(temp, (1, 0, 2)), (100, -1)))
    cv2.waitKey(-1)


def toGif(path, filename):
    imgs = []
    for i in xrange(path.shape[0]):
        img = cv2.cvtColor((path[i] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        imgs.append(np.transpose(img, [2, 0, 1]))
    write_gif(imgs, filename, fps=5)


if __name__ == "__main__":
    space, minkowsky = draw_map((100, 100), 10, 6)
    path = search_path(minkowsky, (5, random.randint(5, 95)), (95, random.randint(5, 95)))
    if len(path) <= 0:
        print "no path is possible!"
    else:
        frames = draw_path(space, 10, path, fixed_length=20)
        plot_path(frames)
        artifact_path = os.path.dirname(os.path.abspath(__file__)) + "/../artifacts/"
        toGif(frames, artifact_path + "sample_path.gif")
