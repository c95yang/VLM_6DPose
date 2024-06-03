import numpy as np
import torch
import torch.nn as nn

from utils.clip_adapter import clip
from utils.pointclip_utils import euler2mat, PCViews


class PCViewsAdapter(PCViews):
    """For creating images from PC based on the view information. Faster as the
    repeated operations are done only once whie initialization.
    """

    def __init__(self):
        super().__init__()
        self.TRANS = -1.4
        self.RESOLUTION = 224
        _views = np.asarray([
            [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, self.TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [0, 0, self.TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [0, 0, self.TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [0, 0, self.TRANS]],
            [[5 * np.pi / 4, -np.pi / 4, np.pi / 2], [0, 0, self.TRANS]],
            [[5 * np.pi / 4, np.pi / 4, np.pi / 2], [0, 0, self.TRANS]],
            [[7 * np.pi / 4, -np.pi / 4, np.pi / 2], [0, 0, self.TRANS]],
            [[7 * np.pi / 4, np.pi / 4, np.pi / 2], [0, 0, self.TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [0, 0, self.TRANS]],
            [[0, np.pi / 2, np.pi / 2], [0, 0, self.TRANS]]])

        self.num_views = 10

        angle = torch.tensor(_views[:, 0, :]).float()
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float()
        self.translation = self.translation.unsqueeze(1)


class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.sync_bn = sync_bn
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        if self.sync_bn:
            # 4d input for BatchNorm2dSync
            x = x.view(s1 * s2, self.feat_size, 1, 1)
            x = self.bn(x)
        else:
            x = x.view(s1 * s2, self.feat_size)
            x = self.bn(x)
        return x.view(s1, s2, s3)


def load_clip_to_cpu():
    backbone_name = 'RN101'
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')

    model = clip.build_model(state_dict or model.state_dict())

    return model


class Adapter(nn.Module):
    """
    Inter-view Adapter
    """

    def __init__(self, num_views):
        super().__init__()

        self.num_views = num_views
        self.in_features = 512
        self.adapter_ratio = 0.6
        self.fusion_init = 0.5
        self.dropout = 0.075

        self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)

        self.global_f = nn.Sequential(
            BatchNormPoint(self.in_features),
            nn.Dropout(self.dropout),
            nn.Flatten(),
            nn.Linear(in_features=self.in_features * self.num_views,
                      out_features=self.in_features),
            nn.BatchNorm1d(self.in_features),
            nn.ReLU(),
            nn.Dropout(self.dropout))

        self.view_f = nn.Sequential(
            nn.Linear(in_features=self.in_features,
                      out_features=self.in_features),
            nn.ReLU(),
            nn.Linear(in_features=self.in_features,
                      out_features=self.in_features * self.num_views),
            nn.ReLU())

    def forward(self, feat):
        img_feat = feat.reshape(-1, self.num_views, self.in_features)
        res_feat = feat.reshape(-1, self.num_views * self.in_features)

        # Global feature
        global_feat = self.global_f(img_feat * self.fusion_ratio.reshape(1, -1, 1))
        # View-wise adapted features
        view_feat = self.view_f(global_feat)

        img_feat = view_feat * self.adapter_ratio + res_feat * (1 - self.adapter_ratio)

        return img_feat