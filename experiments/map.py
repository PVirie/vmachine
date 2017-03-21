import cv2
import numpy as np
from array2gif import write_gif


def create_polynomial_data_matrix(order, num_data=None):

    if num_data is None:
        num_data = order

    step = 1.0 / (num_data - 1)
    X = np.tile(np.arange(0, 1.0 + step, step, dtype=np.float32), (order, 1))
    A = np.power(X.transpose(), np.tile(np.arange(0, order, 1.0, dtype=np.float32), (num_data, 1)))

    return A


def gen_path(size, obj_size, path_length, path_complexity):
    bases = [[0, 0], [0.5 * size[0], 0.1 * size[1]], [0.5 * size[0], 0.9 * size[1]], [1.0 * size[0], 1.0 * size[1]]]

    A_ = np.linalg.pinv(create_polynomial_data_matrix(path_complexity))

    npb = np.asarray(bases, dtype=np.float32)
    px = np.matmul(A_, npb[:, 0:1])
    py = np.matmul(A_, npb[:, 1:2])

    A = create_polynomial_data_matrix(path_complexity, path_length)

    X = np.matmul(A, px)
    Y = np.matmul(A, py)

    path = []
    for i in xrange(path_length):
        path.append((X[i], Y[i]))
    return path


def draw_path(size, obj_size, path):
    canvas = np.zeros((len(path), size[0], size[1]), dtype=np.float32)
    for i in xrange(len(path)):
        cv2.circle(canvas[i], path[i], obj_size / 2, (1.0, 1.0, 1.0), -1)
    return canvas


def plot_path(path):
    temp = np.copy(path)
    temp[:, 0, :] = 1
    temp[:, :, 0] = 1
    cv2.imshow("a", np.reshape(np.transpose(temp, (1, 0, 2)), (100, -1)))
    cv2.waitKey(-1)


def toGif(path):
    imgs = []
    for i in xrange(path.shape[0]):
        img = cv2.cvtColor((path[i] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        imgs.append(np.transpose(img, [2, 0, 1]))
    write_gif(imgs, './move.gif', fps=5)


if __name__ == "__main__":
    path = draw_path((100, 100), 10, gen_path((100, 100), 10, 20, 4))
    plot_path(path)
    # toGif(path)
