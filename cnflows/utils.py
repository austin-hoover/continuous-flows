import numpy as np
import sklearn.datasets
import sklearn.utils
import torch


def antideriv_tanh(x):
    return torch.abs(x) + torch.log(1.0 + torch.exp(-2.0 * torch.abs(x)))


def deriv_tanh(x):
    return 1.0 - torch.pow(torch.tanh(x), 2)


def gen_toy_data(name="moons", n=200, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    if name == "2spirals":
        n = np.sqrt(np.random.rand(n // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(n // 2, 1) * 0.5
        data = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        data += np.random.randn(*data.shape) * 0.1
        data = data.astype("float32")
        return data
    
    elif name == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        data = []
        for i in range(n):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            data.append(point)
        data = np.array(data)
        data /= 1.414
        data = data.astype("float32")
        return data

    elif name == "checkerboard":
        x1 = np.random.rand(n) * 4 - 2
        x2_ = np.random.rand(n) - np.random.randint(0, 2,n) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        data = data.astype("float32")
        return data

    elif name == "circles":
        data = sklearn.datasets.make_circles(n_samples=n, factor=.5, noise=0.08)[0]
        data *= 3.0
        data = data.astype("float32")
        return data

    elif name == "cos":
        x = rng.rand(n) * 5 - 2.5
        y = np.sin(x) * 2.5
        data = np.stack((x, y), 1)
        data = data.astype("float32")
        return data

    elif name == "line":
        x = rng.rand(n) * 5 - 2.5
        data = np.stack((x, x), 1)
        data = data.astype("float32")
        return data

    elif name == "moons":
        data = sklearn.datasets.make_moons(n_samples=n, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2.0 + np.array([-1.0, -0.2])
        return data

    elif name == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class =n // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        data = 2.0 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        data = data.astype("float32")
        return data

    elif name == "rings":
        n_samples4 = n_samples3 = n_samples2 =n // 4
        n_samples1 =n - n_samples4 - n_samples3 - n_samples2
        linspace4 = np.linspace(0.0, 2.0 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0.0, 2.0 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0.0, 2.0 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0.0, 2.0 * np.pi, n_samples1, endpoint=False)
        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25
        data = 3.0 * np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
        ])
        data = data.T
        data = sklearn.utils.shuffle(data, random_state=rng)
        data = data + rng.normal(scale=0.08, size=data.shape)
        data = data.astype("float32")
        return data
        
    elif name == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=n, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5.0
        data = data.astype("float32")
        return data

    else:
        raise ValueError("invalid name")
        