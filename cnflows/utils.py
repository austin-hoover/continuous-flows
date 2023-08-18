import numpy as np
import sklearn.datasets
import sklearn.utils
import torch


def gen_toy_data(name="circles", size=200, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    if name == "2spirals":
        n = np.sqrt(np.random.rand(size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(size // 2, 1) * 0.5
        d1y = +np.sin(n) * n + np.random.rand(size // 2, 1) * 0.5
        data = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        data += np.random.randn(*data.shape) * 0.1
        idx = np.random.permutation(np.arange(data.shape[0]))
        data = data[idx]
        return data

    elif name == "8gaussians":
        scale = 4.0
        centers = [
            (+1.0, +0.0),
            (-1.0, +0.0),
            (+0.0, +1.0),
            (+0.0, -1.0),
            (+np.sqrt(0.5), +np.sqrt(0.5)),
            (+np.sqrt(0.5), -np.sqrt(0.5)),
            (-np.sqrt(0.5), +np.sqrt(0.5)),
            (-np.sqrt(0.5), -np.sqrt(0.5)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        data = []
        for i in range(size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            data.append(point)
        data = np.array(data, dtype="float32")
        data = data / 1.414
        return data

    elif name == "checkerboard":
        x1 = np.random.rand(size) * 4 - 2
        x2_ = np.random.rand(size) - np.random.randint(0, 2, size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2.0
        return data

    elif name == "circles":
        data = sklearn.datasets.make_circles(n_samples=size, factor=0.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3.0
        return data

    elif name == "moons":
        data = sklearn.datasets.make_moons(n_samples=size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2.0 + np.array([-1.0, -0.2])
        return data

    elif name == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        features = rng.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)
        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        return 2.0 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif name == "rings":
        n_samples4 = n_samples3 = n_samples2 = size // 4
        n_samples1 = size - n_samples4 - n_samples3 - n_samples2
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)
        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25
        data = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
        ])
        data = 3.0 * data
        data = data.T
        data = util_shuffle(data, random_state=rng)
        data = data + rng.normal(scale=0.08, size=data.shape)
        return data.astype("float32")

    elif name == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data = data / 5.0
        return data

    elif name == "line":
        x = rng.rand(size) * 5.0 - 2.5
        y = x
        return np.stack((x, y), 1)
        
    elif name == "cos":
        x = rng.rand(size) * 5.0 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
        
    else:
        raise ValueError("Invalid data name.")

        