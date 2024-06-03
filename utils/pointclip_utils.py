import os
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# realistic projection parameters
params = {'maxpoolz': 1, 'maxpoolxy': 7, 'maxpoolpadz': 0, 'maxpoolpadxy': 2,
          'convz': 1, 'convxy': 3, 'convsigmaxy': 3, 'convsigmaz': 1, 'convpadz': 0, 'convpadxy': 1,
          'imgbias': 0., 'depth_bias': 0.2, 'obj_ratio': 0.8, 'bg_clr': 0.0,
          'resolution': 112, 'depth': 8}


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=0, classname=''):
        # assert isinstance(impath, str)
        # assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ModelNet40(Dataset):

    def __init__(self, split='train'):
        self.split = split

        self.dataset_dir = 'data/modelnet40_ply_hdf5_2048'

        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        self.classnames = self.read_classnames(text_file)

        train_data, train_label = self.load_data(os.path.join(self.dataset_dir, 'train_files.txt'))
        test_data, test_label = self.load_data(os.path.join(self.dataset_dir, 'test_files.txt'))

        if split == 'train':
            self.data = self.read_data(self.classnames, train_data, train_label)
        else:
            self.data = self.read_data(self.classnames, test_data, test_label)

    def load_data(self, data_path):
        all_data = []
        all_label = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                all_data.append(data)
                all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)

        return all_data, all_label

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                classname = line.strip()
                classnames[i] = classname
        return classnames

    def read_data(self, classnames, datas, labels):
        items = []

        for i, data in enumerate(datas):
            label = int(labels[i])
            classname = classnames[label]

            item = Datum(
                impath=data,
                label=label,
                classname=classname
            )
            items.append(item)

        return items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath
        }

        pointcloud = item.impath[: 1024]

        if self.split == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        output['img'] = pointcloud

        return output


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """

    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach() * 0
    one = zero.detach() + 1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat


class PCViews:
    """For creating images from PC based on the view information. Faster as the
    repeated operations are done only once whie initialization.
    """

    def __init__(self):
        self.TRANS = -1.6
        self.RESOLUTION = 128
        _views = np.asarray([
            [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, self.TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [0, 0, self.TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [0, 0, self.TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [0, 0, self.TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [0, 0, self.TRANS]],
            [[0, np.pi / 2, np.pi / 2], [0, 0, self.TRANS]]]
        )

        self.num_views = 6

        angle = torch.tensor(_views[:, 0, :]).float()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float()
        self.translation = self.translation.unsqueeze(1)

    def get_img(self, points):
        """Get image based on the prespecified specifications.

        Args:
            points (torch.tensor): of size [B, _, 3]
        Returns:
            img (torch.tensor): of size [B * self.num_views, RESOLUTION,
                RESOLUTION]
        """
        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))

        img = points2depth(
            points=_points,
            image_height=self.RESOLUTION,
            image_width=self.RESOLUTION,
            size_x=1,
            size_y=1,
        )
        return img

    @staticmethod
    def point_transform(points, rot_mat, translation):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = points - translation
        return points


def distribute(depth, _x, _y, size_x, size_y, image_height, image_width):
    """
    Distributes the depth associated with each point to the discrete coordinates (image_height, image_width) in a region
    of size (size_x, size_y).
    :param depth:
    :param _x:
    :param _y:
    :param size_x:
    :param size_y:
    :param image_height:
    :param image_width:
    :return:
    """

    assert size_x % 2 == 0 or size_x == 1
    assert size_y % 2 == 0 or size_y == 1
    batch, _ = depth.size()
    epsilon = torch.tensor([1e-12], requires_grad=False, device=depth.device)
    _i = torch.linspace(-size_x / 2, (size_x / 2) - 1, size_x, requires_grad=False, device=depth.device)
    _j = torch.linspace(-size_y / 2, (size_y / 2) - 1, size_y, requires_grad=False, device=depth.device)

    extended_x = _x.unsqueeze(2).repeat([1, 1, size_x]) + _i  # [batch, num_points, size_x]
    extended_y = _y.unsqueeze(2).repeat([1, 1, size_y]) + _j  # [batch, num_points, size_y]

    extended_x = extended_x.unsqueeze(3).repeat([1, 1, 1, size_y])  # [batch, num_points, size_x, size_y]
    extended_y = extended_y.unsqueeze(2).repeat([1, 1, size_x, 1])  # [batch, num_points, size_x, size_y]

    extended_x.ceil_()
    extended_y.ceil_()

    value = depth.unsqueeze(2).unsqueeze(3).repeat([1, 1, size_x, size_y])  # [batch, num_points, size_x, size_y]

    # all points that will be finally used
    masked_points = ((extended_x >= 0)
                     * (extended_x <= image_height - 1)
                     * (extended_y >= 0)
                     * (extended_y <= image_width - 1)
                     * (value >= 0))

    true_extended_x = extended_x
    true_extended_y = extended_y

    # to prevent error
    extended_x = (extended_x % image_height)
    extended_y = (extended_y % image_width)

    # [batch, num_points, size_x, size_y]
    distance = torch.abs((extended_x - _x.unsqueeze(2).unsqueeze(3))
                         * (extended_y - _y.unsqueeze(2).unsqueeze(3)))
    weight = (masked_points.float()
              * (1 / (value + epsilon)))  # [batch, num_points, size_x, size_y]
    weighted_value = value * weight

    weight = weight.view([batch, -1])
    weighted_value = weighted_value.view([batch, -1])

    coordinates = (extended_x.view([batch, -1]) * image_width) + extended_y.view(
        [batch, -1])
    coord_max = image_height * image_width
    true_coordinates = (true_extended_x.view([batch, -1]) * image_width) + true_extended_y.view(
        [batch, -1])
    true_coordinates[~masked_points.view([batch, -1])] = coord_max
    weight_scattered = torch.zeros(
        [batch, image_width * image_height],
        device=depth.device).scatter_add(1, coordinates.long(), weight)

    masked_zero_weight_scattered = (weight_scattered == 0.0)
    weight_scattered += masked_zero_weight_scattered.float()

    weighed_value_scattered = torch.zeros(
        [batch, image_width * image_height],
        device=depth.device).scatter_add(1, coordinates.long(), weighted_value)

    return weighed_value_scattered, weight_scattered


def points2depth(points, image_height, image_width, size_x=4, size_y=4):
    """
    :param points: [B, num_points, 3]
    :param image_width:
    :param image_height:
    :param size_x:
    :param size_y:
    :return:
        depth_recovered: [B, image_width, image_height]
    """

    epsilon = torch.tensor([1e-12], requires_grad=False, device=points.device)
    # epsilon not needed, kept here to ensure exact replication of old version
    coord_x = (points[:, :, 0] / (points[:, :, 2] + epsilon)) * (image_width / image_height)  # [batch, num_points]
    coord_y = (points[:, :, 1] / (points[:, :, 2] + epsilon))  # [batch, num_points]

    batch, total_points, _ = points.size()
    depth = points[:, :, 2]  # [batch, num_points]
    # pdb.set_trace()
    _x = ((coord_x + 1) * image_height) / 2
    _y = ((coord_y + 1) * image_width) / 2

    weighed_value_scattered, weight_scattered = distribute(
        depth=depth,
        _x=_x,
        _y=_y,
        size_x=size_x,
        size_y=size_y,
        image_height=image_height,
        image_width=image_width)

    depth_recovered = (weighed_value_scattered / weight_scattered).view([
        batch, image_height, image_width
    ])

    return depth_recovered


def points2grid(points, resolution=params['resolution'], depth=params['depth']):
    from torch_scatter import scatter
    """Quantize each point cloud to a 3D grid.
    Args:
        points (torch.tensor): of size [B, _, 3]
    Returns:
        grid (torch.tensor): of size [B * self.num_views, depth, resolution, resolution]
    """
    batch, pnum, _ = points.shape

    pmax, pmin = points.max(dim=1)[0], points.min(dim=1)[0]
    pcent = (pmax + pmin) / 2
    pcent = pcent[:, None, :]
    prange = (pmax - pmin).max(dim=-1)[0][:, None, None]
    points = (points - pcent) / prange * 2.
    points[:, :, :2] = points[:, :, :2] * params['obj_ratio']

    depth_bias = params['depth_bias']
    _x = (points[:, :, 0] + 1) / 2 * resolution
    _y = (points[:, :, 1] + 1) / 2 * resolution
    _z = ((points[:, :, 2] + 1) / 2 + depth_bias) / (1 + depth_bias) * (depth - 2)

    _x.ceil_()
    _y.ceil_()
    z_int = _z.ceil()

    _x = torch.clip(_x, 1, resolution - 2)
    _y = torch.clip(_y, 1, resolution - 2)
    _z = torch.clip(_z, 1, depth - 2)

    coordinates = z_int * resolution * resolution + _y * resolution + _x
    device = points.device if points.is_cuda else torch.device('cpu')
    grid = torch.ones([batch, depth, resolution, resolution], device=device).view(batch, -1) * params['bg_clr']
    grid = scatter(_z.to(device), coordinates.long().to(device), dim=1, out=grid, reduce="max")
    grid = grid.reshape((batch, depth, resolution, resolution)).permute((0, 1, 3, 2))

    return grid


class Grid2Image(nn.Module):
    """A pytorch implementation to turn 3D grid to 2D image.
       Maxpool: densifying the grid
       Convolution: smoothing via Gaussian
       Maximize: squeezing the depth channel
    """

    def __init__(self):
        super().__init__()
        torch.backends.cudnn.benchmark = False

        self.maxpool = nn.MaxPool3d((params['maxpoolz'], params['maxpoolxy'], params['maxpoolxy']),
                                    stride=1, padding=(params['maxpoolpadz'], params['maxpoolpadxy'],
                                                       params['maxpoolpadxy']))
        self.conv = torch.nn.Conv3d(1, 1, kernel_size=(params['convz'], params['convxy'], params['convxy']),
                                    stride=1, padding=(params['convpadz'], params['convpadxy'], params['convpadxy']),
                                    bias=True)
        kn3d = get3DGaussianKernel(params['convxy'], params['convz'], sigma=params['convsigmaxy'], zsigma=params['convsigmaz'])
        self.conv.weight.data = torch.Tensor(kn3d).repeat(1, 1, 1, 1, 1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.maxpool(x.unsqueeze(1))
        x = self.conv(x)
        img = torch.max(x, dim=2)[0]
        img = img / torch.max(torch.max(img, dim=-1)[0], dim=-1)[0][:, :, None, None]
        img = 1 - img
        img = img.repeat(1, 3, 1, 1)
        return img


class Realistic_Projection:
    """For creating images from PC based on the view information.
    """

    def __init__(self):
        self.TRANS = -1.6
        self.RESOLUTION = 128
        _views = np.asarray([
            [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, self.TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, self.TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, self.TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, self.TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, self.TRANS]],
            [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, self.TRANS]],
        ])

        # adding some bias to the view angle to reveal more surface
        _views_bias = np.asarray([
            [[0, np.pi / 9, 0], [-0.5, 0, self.TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, self.TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, self.TRANS]],
            [[0, np.pi / 9, 0], [-0.5, 0, self.TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, self.TRANS]],
            [[0, np.pi / 15, 0], [-0.5, 0, self.TRANS]],
        ])

        self.num_views = 6

        angle = torch.tensor(_views[:, 0, :]).float()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        angle2 = torch.tensor(_views_bias[:, 0, :]).float()
        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)

        self.translation = torch.tensor(_views[:, 1, :]).float()
        self.translation = self.translation.unsqueeze(1)

        self.grid2image = Grid2Image()

    def get_img(self, points):
        b, _, _ = points.shape
        v = self.translation.shape[0]

        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            rot_mat2=self.rot_mat2.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))

        grid = points2grid(points=_points, resolution=params['resolution'], depth=params['depth']).squeeze()

        img = self.grid2image(grid)
        return img

    @staticmethod
    def point_transform(points, rot_mat, rot_mat2, translation):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :param rot_mat2: [batch, 3]
        :param translation: [batch, 1, 3]
        :return:
        """
        rot_mat = rot_mat.to(points.device)
        rot_mat2 = rot_mat2.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = torch.matmul(points, rot_mat2)
        points = points - translation
        return points


def get2DGaussianKernel(ksize, sigma=0):
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()
    return kernel


def get3DGaussianKernel(ksize, depth, sigma=2, zsigma=2):
    kernel2d = get2DGaussianKernel(ksize, sigma)
    zs = (np.arange(depth, dtype=np.float32) - depth // 2)
    zkernel = np.exp(-(zs ** 2) / (2 * zsigma ** 2))
    kernel3d = np.repeat(kernel2d[None, :, :], depth, axis=0) * zkernel[:, None, None]
    kernel3d = kernel3d / torch.sum(kernel3d)
    return kernel3d