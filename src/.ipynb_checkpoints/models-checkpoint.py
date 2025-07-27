#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

# 新增CLIP相关导入
import clip
from PIL import Image

try:
    from layers import *
except:
    from src.layers import *


class GAT(nn.Module):
    def __init__(
        self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag
    ):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                MultiHeadGraphAttention(
                    n_heads[i],
                    f_in,
                    n_units[i + 1],
                    attn_dropout,
                    diag,
                    nn.init.ones_,
                    False,
                )
            )

    def forward(self, x, adj):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, adj)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs"""
    return im.mm(s.t())


def l2norm(X):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X) + 1e-8
    X = torch.div(X, a)
    return X


class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(
            torch.ones((self.modal_num, 1)), requires_grad=self.requires_grad
        )

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [
            weight_norm[idx] * F.normalize(embs[idx])
            for idx in range(self.modal_num)
            if embs[idx] is not None
        ]
        joint_emb = torch.cat(embs, dim=1)
        return joint_emb


class MultiModalFusionNew_allmodal(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(
            torch.ones((self.modal_num, 1)), requires_grad=self.requires_grad
        )

        self.linear_0 = nn.Linear(100, 600)
        self.linear_1 = nn.Linear(100, 600)
        self.linear_2 = nn.Linear(100, 600)
        self.linear_3 = nn.Linear(300, 600)
        self.linear_4 = nn.Linear(100, 600)  # 新增：CLIP文本特征
        self.linear = nn.Linear(600, 600)
        self.v = nn.Linear(600, 1, bias=False)

        self.LN_pre = nn.LayerNorm(600)

    def forward(self, embs):
        assert len(embs) == self.modal_num

        emb_list = []

        if embs[0] is not None:  # img
            emb_list.append(self.linear_0(embs[0]).unsqueeze(1))
        if embs[1] is not None:  # attr
            emb_list.append(self.linear_1(embs[1]).unsqueeze(1))
        if embs[2] is not None:  # rel
            emb_list.append(self.linear_2(embs[2]).unsqueeze(1))
        if embs[3] is not None:  # graph
            emb_list.append(self.linear_3(embs[3]).unsqueeze(1))
        if embs[4] is not None:  # text (CLIP)
            emb_list.append(self.linear_4(embs[4]).unsqueeze(1))
        if embs[5] is not None:  # name
            emb_list.append(self.linear_0(embs[5]).unsqueeze(1))
        if embs[6] is not None:  # char
            emb_list.append(self.linear_0(embs[6]).unsqueeze(1))
            
        if len(emb_list) == 0:
            # 如果没有特征，返回零向量
            return torch.zeros((embs[0].size(0) if embs[0] is not None else 1, 600)).to(
                next(self.parameters()).device
            )
            
        new_embs = torch.cat(emb_list, dim=1)  # [n, num_modals, e]
        new_embs = self.LN_pre(new_embs)
        s = self.v(torch.tanh(self.linear(new_embs)))  # [n, num_modals, 1]
        a = torch.softmax(s, dim=-1)
        joint_emb_1 = torch.matmul(a.transpose(-1, -2), new_embs).squeeze(1)  # [n, e]
        joint_emb = joint_emb_1
        return joint_emb


class IBMultiModal(nn.Module):
    def __init__(
        self,
        args,
        ent_num,
        img_feature_dim,
        char_feature_dim=None,
        use_project_head=False,
    ):
        super(IBMultiModal, self).__init__()

        self.args = args
        attr_dim = self.args.attr_dim
        img_dim = self.args.img_dim
        char_dim = self.args.char_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.use_project_head = use_project_head

        # === CLIP集成 ===
        self.use_clip = getattr(args, 'use_clip', False)
        self.freeze_clip = getattr(args, 'freeze_clip', True)
        
        if self.use_clip:
            print("Loading CLIP model...")
            self.clip_model, self.clip_preprocess = clip.load(
                getattr(args, 'clip_model', "ViT-B/32"), 
                device=args.device if hasattr(args, 'device') else 'cuda'
            )
            
            # 冻结CLIP参数
            if self.freeze_clip:
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                print("CLIP parameters frozen")
            
            # CLIP特征维度
            self.clip_dim = 512  # ViT-B/32的特征维度
            
            # CLIP特征投影层
            self.clip_img_proj = nn.Linear(self.clip_dim, img_dim)
            self.clip_text_proj = nn.Linear(self.clip_dim, attr_dim)
            
            # CLIP文本特征VIB层
            self.use_text_vib = getattr(args, 'use_text_vib', False)
            if self.use_text_vib:
                self.text_fc_mu = nn.Linear(attr_dim, attr_dim)
                self.text_fc_std = nn.Linear(attr_dim, attr_dim)

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        # 关系特征层
        self.rel_fc = nn.Linear(1000, attr_dim)
        self.rel_fc_mu = nn.Linear(attr_dim, attr_dim)
        self.rel_fc_std = nn.Linear(attr_dim, attr_dim)
        self.rel_fc_d1 = nn.Linear(attr_dim, attr_dim)
        self.rel_fc_d2 = nn.Linear(attr_dim, attr_dim)

        # 属性特征层
        self.att_fc = nn.Linear(1000, attr_dim)
        self.att_fc_mu = nn.Linear(attr_dim, attr_dim)
        self.att_fc_std = nn.Linear(attr_dim, attr_dim)
        self.att_fc_d1 = nn.Linear(attr_dim, attr_dim)
        self.att_fc_d2 = nn.Linear(attr_dim, attr_dim)

        # 图像特征层（保留作为fallback）
        self.img_fc = nn.Linear(img_feature_dim, img_dim)
        self.img_fc_mu = nn.Linear(img_dim, img_dim)
        self.img_fc_std = nn.Linear(img_dim, img_dim)
        self.img_fc_d1 = nn.Linear(img_dim, img_dim)
        self.img_fc_d2 = nn.Linear(img_dim, img_dim)

        # 联合特征层
        joint_dim = 600
        self.joint_fc = nn.Linear(joint_dim, joint_dim)
        self.joint_fc_mu = nn.Linear(joint_dim, joint_dim)
        self.joint_fc_std = nn.Linear(joint_dim, joint_dim)
        self.joint_fc_d1 = nn.Linear(joint_dim, joint_dim)
        self.joint_fc_d2 = nn.Linear(joint_dim, joint_dim)

        # VIB配置
        use_graph_vib = self.args.use_graph_vib
        use_attr_vib = self.args.use_attr_vib
        use_img_vib = self.args.use_img_vib
        use_rel_vib = self.args.use_rel_vib

        no_diag = self.args.no_diag
        diag = False if no_diag else True

        self.use_graph_vib = use_graph_vib
        self.use_attr_vib = use_attr_vib
        self.use_img_vib = use_img_vib
        self.use_rel_vib = use_rel_vib
        self.use_joint_vib = self.args.use_joint_vib

        # 名称和字符特征层
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)

        self.kld_loss = 0
        self.gph_layer_norm_mu = nn.LayerNorm(self.input_dim, elementwise_affine=True)
        self.gph_layer_norm_std = nn.LayerNorm(self.input_dim, elementwise_affine=True)

        # 图结构编码器
        if self.args.structure_encoder == "gcn":
            if self.use_graph_vib:
                self.cross_graph_model_mu = GCN(
                    self.n_units[0],
                    self.n_units[1],
                    self.n_units[2],
                    dropout=self.args.dropout,
                )
                self.cross_graph_model_std = GCN(
                    self.n_units[0],
                    self.n_units[1],
                    self.n_units[2],
                    dropout=self.args.dropout,
                )
            else:
                self.cross_graph_model = GCN(
                    self.n_units[0],
                    self.n_units[1],
                    self.n_units[2],
                    dropout=self.args.dropout,
                )
        elif self.args.structure_encoder == "gat":
            if self.use_graph_vib:
                self.cross_graph_model_mu = GAT(
                    n_units=self.n_units,
                    n_heads=self.n_heads,
                    dropout=args.dropout,
                    attn_dropout=args.attn_dropout,
                    instance_normalization=self.args.instance_normalization,
                    diag=diag,
                )
                self.cross_graph_model_std = GAT(
                    n_units=self.n_units,
                    n_heads=self.n_heads,
                    dropout=args.dropout,
                    attn_dropout=args.attn_dropout,
                    instance_normalization=self.args.instance_normalization,
                    diag=diag,
                )
            else:
                self.cross_graph_model = GAT(
                    n_units=self.n_units,
                    n_heads=self.n_heads,
                    dropout=args.dropout,
                    attn_dropout=args.attn_dropout,
                    instance_normalization=self.args.instance_normalization,
                    diag=True,
                )

        # 投影头
        if self.use_project_head:
            self.img_pro = ProjectionHead(img_dim, img_dim, img_dim, dropout)
            self.att_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.rel_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.gph_pro = ProjectionHead(
                self.n_units[2], self.n_units[2], self.n_units[2], dropout
            )
            if self.use_clip:
                self.text_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)

        # 融合模块
        modal_num = self.args.inner_view_num
        if self.use_clip:
            modal_num += 1  # 为CLIP文本特征增加一个模态
            
        if self.args.fusion_id == 1:
            self.fusion = MultiModalFusion(
                modal_num=modal_num, with_weight=self.args.with_weight
            )
        elif self.args.fusion_id == 2:
            self.fusion = MultiModalFusionNew_allmodal(
                modal_num=modal_num, with_weight=self.args.with_weight
            )

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        from torch.distributions.kl import kl_divergence
        from torch.distributions import Normal

        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_1, 0.4)))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_2, 0.4)))
        mu_1_fixed = mu_1.clone()
        sigma_1_fixed = sigma_1.clone()
        mu_1_fixed[torch.isnan(mu_1_fixed)] = 0
        mu_1_fixed[torch.isinf(mu_1_fixed)] = torch.max(
            mu_1_fixed[~torch.isinf(mu_1_fixed)]
        )
        sigma_1_fixed[torch.isnan(sigma_1_fixed)] = 1
        sigma_1_fixed[torch.isinf(sigma_1_fixed)] = torch.max(
            sigma_1_fixed[~torch.isinf(sigma_1_fixed)]
        )
        sigma_1_fixed[sigma_1_fixed <= 0] = 1
        q_target = Normal(mu_1_fixed, sigma_1_fixed)

        mu_2_fixed = mu_2.clone()
        sigma_2_fixed = sigma_2.clone()
        mu_2_fixed[torch.isnan(mu_2_fixed)] = 0
        mu_2_fixed[torch.isinf(mu_2_fixed)] = torch.max(
            mu_2_fixed[~torch.isinf(mu_2_fixed)]
        )
        sigma_2_fixed[torch.isnan(sigma_2_fixed)] = 1
        sigma_2_fixed[torch.isinf(sigma_2_fixed)] = torch.max(
            sigma_2_fixed[~torch.isinf(sigma_2_fixed)]
        )
        sigma_2_fixed[sigma_2_fixed <= 0] = 1
        q_context = Normal(mu_2_fixed, sigma_2_fixed)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return kl

    def encode_clip_features(self, raw_images=None, text_descriptions=None):
        """
        使用CLIP编码图像和文本特征
        """
        clip_img_emb = None
        clip_text_emb = None
        
        if self.use_clip:
            # 编码图像
            if raw_images is not None and len(raw_images) > 0:
                try:
                    with torch.no_grad():
                        clip_img_features = self.clip_model.encode_image(raw_images)
                    clip_img_emb = self.clip_img_proj(clip_img_features.float())
                except Exception as e:
                    print(f"Error encoding images with CLIP: {e}")
                    clip_img_emb = None
            
            # 编码文本
            if text_descriptions is not None and len(text_descriptions) > 0:
                try:
                    with torch.no_grad():
                        text_tokens = clip.tokenize(text_descriptions, truncate=True).to(
                            next(self.parameters()).device
                        )
                        clip_text_features = self.clip_model.encode_text(text_tokens)
                    clip_text_emb = self.clip_text_proj(clip_text_features.float())
                except Exception as e:
                    print(f"Error encoding text with CLIP: {e}")
                    clip_text_emb = None
        
        return clip_img_emb, clip_text_emb

    def forward(
        self,
        input_idx,
        adj,
        img_features=None,
        rel_features=None,
        att_features=None,
        name_features=None,
        char_features=None,
        raw_images=None,
        text_descriptions=None,
    ):
        # === 图结构特征处理 ===
        if self.args.w_gcn:
            if self.use_graph_vib:
                if self.args.structure_encoder == "gat":
                    gph_emb_mu = self.cross_graph_model_mu(
                        self.entity_emb(input_idx), adj
                    )
                    mu = self.gph_layer_norm_mu(gph_emb_mu)

                    gph_emb_std = self.cross_graph_model_std(
                        self.entity_emb(input_idx), adj
                    )
                    std = F.elu(gph_emb_std)
                    eps = torch.randn_like(std)
                    gph_emb = mu + eps * std
                    gph_kld_loss = self._kld_gauss(
                        mu, std, torch.zeros_like(mu), torch.ones_like(std)
                    )
                    self.kld_loss = gph_kld_loss
                else:
                    mu = self.cross_graph_model_mu(self.entity_emb(input_idx), adj)
                    logstd = self.cross_graph_model_std(self.entity_emb(input_idx), adj)
                    eps = torch.randn_like(mu)
                    gph_emb = mu + eps * torch.exp(logstd)
                    gph_kld_loss = self._kld_gauss(
                        mu, logstd, torch.zeros_like(mu), torch.ones_like(logstd)
                    )
                    self.kld_loss = gph_kld_loss
            else:
                gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
        else:
            gph_emb = None

        # === CLIP特征提取 ===
        clip_img_emb, clip_text_emb = self.encode_clip_features(raw_images, text_descriptions)

        # === 图像特征处理 ===
        if self.args.w_img:
            if self.use_clip and clip_img_emb is not None:
                # 使用CLIP图像特征
                img_emb = clip_img_emb
            else:
                # 使用传统图像特征作为fallback
                img_emb = self.img_fc(img_features)
            
            # VIB处理
            if self.use_img_vib:
                img_emb_h = F.relu(img_emb)
                mu = self.img_fc_mu(img_emb_h)
                logvar = self.img_fc_std(img_emb_h)
                std = torch.exp(0.5 * logvar)
                eps = torch.rand_like(std)
                img_emb = mu + eps * std
                img_kld_loss = self._kld_gauss(
                    mu, std, torch.zeros_like(mu), torch.ones_like(std)
                )
                self.img_kld_loss = img_kld_loss
        else:
            img_emb = None

        # === CLIP文本特征处理 ===
        text_emb = None
        if self.use_clip and clip_text_emb is not None:
            text_emb = clip_text_emb
            
            # 文本VIB处理
            if self.use_text_vib:
                text_emb_h = F.relu(text_emb)
                mu = self.text_fc_mu(text_emb_h)
                logvar = self.text_fc_std(text_emb_h)
                std = torch.exp(0.5 * logvar)
                eps = torch.rand_like(std)
                text_emb = mu + eps * std
                text_kld_loss = self._kld_gauss(
                    mu, std, torch.zeros_like(mu), torch.ones_like(std)
                )
                self.text_kld_loss = text_kld_loss

        # === 关系特征处理 ===
        if self.args.w_rel:
            if self.use_rel_vib:
                rel_emb = self.rel_fc(rel_features)
                rel_emb_h = F.relu(rel_emb)
                mu = self.rel_fc_mu(rel_emb_h)
                logvar = self.rel_fc_std(rel_emb_h)
                std = torch.exp(0.5 * logvar)
                eps = torch.rand_like(std)
                rel_emb = mu + eps * std
                rel_kld_loss = self._kld_gauss(
                    mu, std, torch.zeros_like(mu), torch.ones_like(std)
                )
                self.rel_kld_loss = rel_kld_loss
            else:
                rel_emb = self.rel_fc(rel_features)
        else:
            rel_emb = None

        # === 属性特征处理 ===
        if self.args.w_attr:
            if self.use_attr_vib:
                att_emb = self.att_fc(att_features)
                att_emb_h = F.relu(att_emb)
                mu = self.att_fc_mu(att_emb_h)
                logvar = self.att_fc_std(att_emb_h)
                std = torch.exp(0.5 * logvar)
                eps = torch.rand_like(std)
                att_emb = mu + eps * std
                attr_kld_loss = self._kld_gauss(
                    mu, std, torch.zeros_like(mu), torch.ones_like(std)
                )
                self.attr_kld_loss = attr_kld_loss
            else:
                att_emb = self.att_fc(att_features)
        else:
            att_emb = None

        # === 名称和字符特征处理 ===
        if self.args.w_name:
            name_emb = self.name_fc(name_features)
        else:
            name_emb = None
        if self.args.w_char:
            char_emb = self.char_fc(char_features)
        else:
            char_emb = None

        # === 投影头处理 ===
        if self.use_project_head:
            if gph_emb is not None:
                gph_emb = self.gph_pro(gph_emb)
            if img_emb is not None:
                img_emb = self.img_pro(img_emb)
            if rel_emb is not None:
                rel_emb = self.rel_pro(rel_emb)
            if att_emb is not None:
                att_emb = self.att_pro(att_emb)
            if text_emb is not None and self.use_clip:
                text_emb = self.text_pro(text_emb)

        # === 多模态融合 ===
        if self.use_clip:
            # 包含CLIP文本特征的融合
            joint_emb = self.fusion([img_emb, att_emb, rel_emb, gph_emb, text_emb, name_emb, char_emb])
        else:
            # 原有融合方式
            joint_emb = self.fusion([img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb])

        # === 联合VIB处理 ===
        if self.use_joint_vib:
            joint_emb = self.joint_fc(joint_emb)
            joint_emb_h = F.relu(joint_emb)
            mu = self.joint_fc_mu(joint_emb_h)
            logvar = self.joint_fc_std(joint_emb_h)
            std = torch.exp(0.5 * logvar)
            eps = torch.rand_like(std)
            joint_emb = mu + eps * std
            joint_kld_loss = self._kld_gauss(
                mu, std, torch.zeros_like(mu), torch.ones_like(std)
            )
            self.joint_kld_loss = joint_kld_loss

        return gph_emb, img_emb, rel_emb, att_emb, text_emb, name_emb, char_emb, joint_emb
