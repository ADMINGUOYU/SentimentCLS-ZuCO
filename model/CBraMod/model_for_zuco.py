import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()

        # in_dim=200: The input dimension for each EEG patch embedding. Input signals are typically resampled to 200Hz and segmented into patches.
        # out_dim=200: The final output dimension of the backbone for each processed patch.
        # d_model=200: The hidden dimension (embedding size) maintained throughout the Transformer layers.
        # dim_feedforward=800: The internal dimension of the Position-wise Feed-Forward Network (FFN) within each Transformer block.
        # seq_len=30: The length of the temporal sequence, likely corresponding to 30-second EEG samples segmented into 1-second patches.
        # n_layer=12: The number of stacked Criss-Cross Transformer blocks in the backbone.
        # nhead=8: The total number of attention heads. In CBraMod, these are often split into S-Attention (spatial) and V-Attention (temporal/vertical) to model dependencies separately. 
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(105 * 1 * 200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(105 * 1 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(105 * 1 * 200, 5 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(5 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )

    def forward(self, x):
        # mock_eeg.shape = (batch_size, num_of_channels, time_segments, points_per_patch)
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out

