#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F

import tracker.pointtrack.models.erfnet as erfnet


def LocationEmbedding(
    f_g: torch.tensor,
    device: torch.device,
    dim_g: int=64,
    wave_len: int=1000,
) -> torch.tensor:

    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    position_mat = torch.cat((cx, cy, w, h), -1)

    feat_range = torch.arange(dim_g / 8, device=device)
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, -1)
    position_mat = position_mat.view(f_g.shape[0], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(f_g.shape[0], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)
    
    return embedding


class BranchedERFNet(nn.Module):
    def __init__(
        self,
        num_classes: list,
        input_channel: int=3,
        encoder=None,
    ) -> None:

        super().__init__()

        if encoder is None:
            self.encoder = erfnet.Encoder(
                sum(num_classes),
                input_channel=input_channel
            )
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    def init_output(
        self,
        n_sigma: int=1,
    ) -> None:

        with torch.no_grad():
            output_conv = self.decoders[0].output_conv

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)

    def forward(
        self,
        input: torch.tensor,
        only_encode: bool=False,
    ) -> torch.tensor:

        if only_encode:
            return self.encoder.forward(
                input,
                predict=True,
            )
        else:
            output = self.encoder(input)

        return torch.cat(
            [
                decoder.forward(output)
                for decoder in self.decoders
            ],
            dim=1,
        )


class PointFeatFuse3P(nn.Module):
    def __init__(
        self,
        num_points: int=250,
        ic: int=7,
        oc: int=64,
        maxpool: bool=True,
    ) -> None:

        super(PointFeatFuse3P, self).__init__()

        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)

        self.t_conv1 = torch.nn.Conv1d(3, 64, 1)
        self.t_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.t_conv3 = torch.nn.Conv1d(128, 128, 1)

        self.conv4 = torch.nn.Conv1d(512, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, oc, 1)

        if maxpool:
            self.pool = torch.nn.MaxPool1d(num_points)
        else:
            self.pool = torch.nn.AvgPool1d(num_points)

        self.num_points = num_points

    def forward(
        self,
        x: torch.tensor,
        emb: torch.tensor,
        t: torch.tensor,
        withInd: bool=False,
    ) -> torch.tensor:

        x = F.leaky_relu(self.conv1(x))
        emb = F.leaky_relu(self.e_conv1(emb))
        t = F.leaky_relu(self.t_conv1(t))

        x = F.leaky_relu(self.conv2(x))
        emb = F.leaky_relu(self.e_conv2(emb))
        t = F.leaky_relu(self.t_conv2(t))

        x = F.leaky_relu(self.conv3(x))
        emb = F.leaky_relu(self.e_conv3(emb))
        t = F.leaky_relu(self.t_conv3(t))

        x1 = F.leaky_relu(self.conv4(torch.cat((x, emb, t), dim=1)))
        x1 = F.leaky_relu(self.conv5(x1))
        x1 = F.leaky_relu(self.conv6(x1))

        if withInd:
            return self.pool(x1).squeeze(-1), torch.max(x1, dim=2)[1]

        return self.pool(x1).squeeze(-1)

class PoseNetFeatOffsetEmb(nn.Module):
    def __init__(
        self,
        num_points: int=250,
        ic: int=7,
        border_ic: int=6,
        output_dim: int=64,
        category: bool=False,
    ) -> None:

        super(PoseNetFeatOffsetEmb, self).__init__()

        self.category = category
        bc = 256
        self.border_points = int(num_points / 3)
        self.num_points = num_points - self.border_points

        self.borderConv = PointFeatFuse3P(
            ic=border_ic,
            oc=bc,
            num_points=self.border_points,
        )

        self.conv1 = torch.nn.Conv1d(2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3_bn = nn.BatchNorm1d(256)

        self.e_conv1 = torch.nn.Conv1d(ic, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)
        self.e_conv1_bn = nn.BatchNorm1d(64)
        self.e_conv2_bn = nn.BatchNorm1d(128)
        self.e_conv3_bn = nn.BatchNorm1d(256)

        self.conv4 = torch.nn.Conv1d(512, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 64, 1)
        self.conv4_bn = nn.BatchNorm1d(256)
        self.conv5_bn = nn.BatchNorm1d(512)

        self.conv7 = torch.nn.Conv1d(512, 256, 1)
        self.conv8 = torch.nn.Conv1d(256, 512, 1)
        self.conv9 = torch.nn.Conv1d(512, 64, 1)
        self.conv7_bn = nn.BatchNorm1d(256)
        self.conv8_bn = nn.BatchNorm1d(512)

        self.conv_weight = torch.nn.Conv1d(128, 1, 1)

        self.last_emb = nn.Sequential(
            nn.Linear(704+bc, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )
        self.ap1 = torch.nn.AvgPool1d(self.num_points)
        self.mp2 = torch.nn.MaxPool1d(self.num_points)

    def forward(
        self,
        inp: torch.tensor,
        borders: torch.tensor,
        spatialEmbs: torch.tensor,
        with_weight: bool=False,
    ) -> torch.tensor:

        x, emb = inp[:,-2:], inp[:,:-2]
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        emb = F.leaky_relu(self.e_conv1_bn(self.e_conv1(emb)))

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        emb = F.leaky_relu(self.e_conv2_bn(self.e_conv2(emb)))

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))          # B,256,N
        emb = F.leaky_relu(self.e_conv3_bn(self.e_conv3(emb)))  # B,256,N

        pointfeat_2 = torch.cat((x, emb), dim=1)

        x1 = F.leaky_relu(self.conv4_bn(self.conv4(pointfeat_2)))
        x1 = F.leaky_relu(self.conv5_bn(self.conv5(x1)))
        x1 = F.leaky_relu(self.conv6(x1))                       # B,64,N
        ap_x1 = self.ap1(x1).squeeze(-1)                        # B,64

        x2 = F.leaky_relu(self.conv7_bn(self.conv7(pointfeat_2)))
        x2 = F.leaky_relu(self.conv8_bn(self.conv8(x2)))
        x2 = F.leaky_relu(self.conv9(x2))                       # B,64,N
        mp_x2 = self.mp2(x2).squeeze(-1)                        # B,64

        weightFeat = self.conv_weight(torch.cat([x1, x2], dim=1))   #B,1,N
        weight = torch.nn.Softmax(2)(weightFeat)
        weight_x3 = (weight.expand_as(pointfeat_2) * pointfeat_2).sum(2)

        if with_weight:
            border_feat, bg_inds = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:], withInd=with_weight)
            x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
            outp = self.last_emb(x)
            return outp, weight, bg_inds
        else:
            border_feat = self.borderConv(borders[:, 3:5], borders[:, :3], borders[:, 5:])

        x = torch.cat([ap_x1, mp_x2, weight_x3, border_feat, spatialEmbs], dim=1)
        outp = self.last_emb(x)

        return outp


class TrackerOffsetEmb(nn.Module):
    def __init__(
        self,
        device: torch.device,
        margin: float=0.3,
        num_points: int=250,
        border_ic: int=6,
        category: bool=False,
        outputD: int=64,
        v23: bool=False,
    ) -> None:

        super().__init__()
        self.point_feat = PoseNetFeatOffsetEmb(
            num_points=num_points,
            ic=3,
            border_ic=border_ic,
            output_dim=outputD,
            category=True,
        )

        self.device = device
        self.num_points = num_points - int(num_points / 3)
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.embedding = LocationEmbedding

    def compute_triplet_loss(
        self,
        inputs: torch.tensor,
        targets: torch.tensor,
    ) -> torch.tensor:

        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        loss = torch.zeros([1], device=self.device)
        if mask.float().unique().shape[0] > 1:
            dist_ap, dist_an = [], []
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            y = torch.ones_like(dist_an)
            loss = self.ranking_loss(dist_an, dist_ap, y).unsqueeze(0)

        return loss

    def forward(
        self,
        points: torch.tensor,
        labels: torch.tensor,
        xyxys: torch.tensor,
        infer: bool=False,
        visualize: bool=False,
    ) -> torch.tensor:

        points, xyxys = points[0], xyxys[0]
        embeds = self.embedding(xyxys, self.device)
        envs = points[:, self.num_points:]
        points = points[:, :self.num_points, :5]

        if infer:
            return self.inference(points, envs, embeds)
        elif visualize:
            embeds, point_weights, bg_inds = self.point_feat(
                points.transpose(2, 1).contiguous(),
                envs.transpose(2, 1).contiguous(),
                embeds,
                with_weight=True,
            )
            return embeds, point_weights, bg_inds
        else:
            embeds = self.point_feat(
                points.transpose(2, 1).contiguous(),
                envs.transpose(2, 1).contiguous(),
                embeds,
            )
            labels = labels[0]
            return self.compute_triplet_loss(embeds, labels)

    def inference(
        self,
        points: torch.tensor,
        envs: torch.tensor,
        embeds: torch.tensor,
    ) -> torch.tensor:

        embeds = self.point_feat(
            points.transpose(2, 1).contiguous(),
            envs.transpose(2, 1).contiguous(),
            embeds
        )
        return embeds
