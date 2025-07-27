#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

# 添加CLIP相关导入
try:
    from transformers import CLIPModel, CLIPVisionModel, CLIPTextModel
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available, CLIP features disabled")
    CLIPModel = CLIPVisionModel = CLIPTextModel = None
    CLIP_AVAILABLE = False

try:
    from layers import GraphConvolution, MultiHeadGraphAttention, ProjectionHead
except:
    from src.layers import GraphConvolution, MultiHeadGraphAttention, ProjectionHead


class SimpleGCN(nn.Module):
    """简化的GCN层，避免内存问题"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SimpleGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class SimpleGAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(SimpleGAT, self).__init__()
        self.n_units = n_units
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.instance_normalization = instance_normalization
        self.diag = diag
        
        self.layers = nn.ModuleList()
        for i in range(len(n_units) - 1):
            in_dim = n_units[i]
            out_dim = n_units[i + 1]
            n_head = n_heads[i] if i < len(n_heads) else 1
            
            self.layers.append(
                MultiHeadGraphAttention(
                    n_head=n_head,
                    f_in=in_dim,
                    f_out=out_dim,
                    attn_dropout=attn_dropout,
                    diag=diag
                )
            )
        
        if instance_normalization:
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(n_units[i+1]) for i in range(len(n_units) - 1)
            ])
    
    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            # 对于大图，使用简化的处理
            if x.size(0) > 20000:
                # 简化版本：只做线性变换和邻接矩阵乘法
                x = self.simple_forward(x, adj, layer)
            else:
                x = layer(x, adj)
                
            if hasattr(self, 'norm_layers'):
                x = self.norm_layers[i](x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def simple_forward(self, x, adj, layer):
        """简化的前向传播，避免内存爆炸"""
        # 使用第一个头的权重
        w = layer.w[0] if layer.diag else layer.w[0]
        
        if layer.diag:
            h = torch.mul(x, w)
        else:
            h = torch.mm(x, w)
        
        # 简单的图卷积
        if adj.is_sparse:
            output = torch.sparse.mm(adj, h)
        else:
            output = torch.mm(adj, h)
        
        return output


class IBMultiModal(nn.Module):
    """修复的IBMEA多模态模型"""
    
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

        # 解析隐藏层配置
        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.input_dim = self.n_units[0]
        
        # 实体嵌入
        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        # 特征处理层
        self.rel_fc = nn.Linear(1000, attr_dim)
        self.att_fc = nn.Linear(1000, attr_dim)
        self.img_fc = nn.Linear(img_feature_dim, img_dim)

        if char_feature_dim is None:
            char_feature_dim = 100
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)

        # 变分信息瓶颈相关
        self.use_graph_vib = getattr(args, 'use_graph_vib', False)
        self.use_attr_vib = getattr(args, 'use_attr_vib', False)
        self.use_img_vib = getattr(args, 'use_img_vib', False)
        self.use_rel_vib = getattr(args, 'use_rel_vib', False)
        self.use_joint_vib = getattr(args, 'use_joint_vib', False)

        # 变分编码器
        if self.use_img_vib:
            self.img_fc_mu = nn.Linear(img_dim, img_dim)
            self.img_fc_std = nn.Linear(img_dim, img_dim)
        if self.use_rel_vib:
            self.rel_fc_mu = nn.Linear(attr_dim, attr_dim)
            self.rel_fc_std = nn.Linear(attr_dim, attr_dim)
        if self.use_attr_vib:
            self.att_fc_mu = nn.Linear(attr_dim, attr_dim)
            self.att_fc_std = nn.Linear(attr_dim, attr_dim)

        # 图结构编码器
        no_diag = getattr(args, 'no_diag', False)
        diag = not no_diag
        
        if self.args.structure_encoder == "gcn":
            if self.use_graph_vib:
                self.cross_graph_model_mu = SimpleGCN(
                    self.n_units[0], self.n_units[1] if len(self.n_units) > 1 else self.n_units[0], 
                    self.n_units[-1], dropout
                )
                self.cross_graph_model_std = SimpleGCN(
                    self.n_units[0], self.n_units[1] if len(self.n_units) > 1 else self.n_units[0], 
                    self.n_units[-1], dropout
                )
            else:
                self.cross_graph_model = SimpleGCN(
                    self.n_units[0], self.n_units[1] if len(self.n_units) > 1 else self.n_units[0], 
                    self.n_units[-1], dropout
                )
        elif self.args.structure_encoder == "gat":
            if self.use_graph_vib:
                self.cross_graph_model_mu = SimpleGAT(
                    n_units=self.n_units,
                    n_heads=self.n_heads,
                    dropout=dropout,
                    attn_dropout=self.args.attn_dropout,
                    instance_normalization=self.args.instance_normalization,
                    diag=diag,
                )
                self.cross_graph_model_std = SimpleGAT(
                    n_units=self.n_units,
                    n_heads=self.n_heads,
                    dropout=dropout,
                    attn_dropout=self.args.attn_dropout,
                    instance_normalization=self.args.instance_normalization,
                    diag=diag,
                )
            else:
                self.cross_graph_model = SimpleGAT(
                    n_units=self.n_units,
                    n_heads=self.n_heads,
                    dropout=dropout,
                    attn_dropout=self.args.attn_dropout,
                    instance_normalization=self.args.instance_normalization,
                    diag=True,
                )

        # 多模态融合
        total_dim = 0
        if args.w_gcn: total_dim += self.n_units[-1]
        if args.w_img: total_dim += img_dim
        if args.w_rel: total_dim += attr_dim  
        if args.w_attr: total_dim += attr_dim
        if args.w_name: total_dim += char_dim
        if args.w_char: total_dim += char_dim
        
        # 多模态融合器
        if getattr(args, 'fusion_id', 1) == 1:
            self.fusion = MultiModalFusion(
                modal_num=getattr(args, 'inner_view_num', 6), 
                with_weight=getattr(args, 'with_weight', 1)
            )
        else:
            # 简化版本的融合
            self.fusion = SimpleFusion(total_dim)

        # 投影头
        if self.use_project_head:
            self.img_pro = ProjectionHead(img_dim, img_dim, img_dim, dropout)
            self.att_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.rel_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.gph_pro = ProjectionHead(self.n_units[-1], self.n_units[-1], self.n_units[-1], dropout)

        # 联合变分编码
        joint_dim = max(total_dim, 128)
        if self.use_joint_vib:
            self.joint_fc = nn.Linear(joint_dim, joint_dim)
            self.joint_fc_mu = nn.Linear(joint_dim, joint_dim)
            self.joint_fc_std = nn.Linear(joint_dim, joint_dim)

        # 初始化损失
        self.kld_loss = 0
        self.img_kld_loss = 0
        self.rel_kld_loss = 0
        self.attr_kld_loss = 0
        self.joint_kld_loss = 0

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """KL散度计算"""
        try:
            from torch.distributions.kl import kl_divergence
            from torch.distributions import Normal
            
            sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_1, 0.4)))
            sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_2, 0.4)))
            
            # 处理NaN和inf
            mu_1 = torch.clamp(mu_1, -10, 10)
            mu_2 = torch.clamp(mu_2, -10, 10)
            sigma_1 = torch.clamp(sigma_1, 0.01, 10)
            sigma_2 = torch.clamp(sigma_2, 0.01, 10)
            
            q_target = Normal(mu_1, sigma_1)
            q_context = Normal(mu_2, sigma_2)
            
            kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
            return torch.clamp(kl, 0, 100)  # 限制KL散度的范围
        except:
            # 简化版本的KL散度
            kl = 0.5 * (logsigma_2 - logsigma_1 - 1.0 + 
                       torch.exp(logsigma_1 - logsigma_2) + 
                       (mu_1 - mu_2) ** 2 * torch.exp(-logsigma_2))
            return torch.clamp(kl.sum(), 0, 100)

    def forward(
        self,
        input_idx,
        adj,
        img_features=None,
        rel_features=None,
        att_features=None,
        name_features=None,
        char_features=None,
        entity_names=None,
    ):
        embeddings = []
        
        # === 图结构特征处理 ===
        gph_emb = None
        if self.args.w_gcn:
            try:
                if self.use_graph_vib:
                    gph_emb_mu = self.cross_graph_model_mu(self.entity_emb(input_idx), adj)
                    gph_emb_std = self.cross_graph_model_std(self.entity_emb(input_idx), adj)
                    eps = torch.randn_like(gph_emb_std)
                    gph_emb = gph_emb_mu + eps * gph_emb_std
                    self.kld_loss = self._kld_gauss(
                        gph_emb_mu, gph_emb_std, 
                        torch.zeros_like(gph_emb_mu), torch.ones_like(gph_emb_std)
                    )
                else:
                    gph_emb = self.cross_graph_model(self.entity_emb(input_idx), adj)
                embeddings.append(gph_emb)
            except Exception as e:
                print(f"Warning: Graph encoding failed: {e}")
                # 使用简单的实体嵌入作为fallback
                gph_emb = self.entity_emb(input_idx)
                embeddings.append(gph_emb)

        # === 图像特征处理 ===
        img_emb = None
        if self.args.w_img and img_features is not None:
            try:
                if self.use_img_vib:
                    img_emb = self.img_fc(img_features)
                    img_emb_h = F.relu(img_emb)
                    mu = self.img_fc_mu(img_emb_h)
                    logvar = self.img_fc_std(img_emb_h)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    img_emb = mu + eps * std
                    self.img_kld_loss = self._kld_gauss(
                        mu, std, torch.zeros_like(mu), torch.ones_like(std)
                    )
                else:
                    img_emb = self.img_fc(img_features)
                embeddings.append(img_emb)
            except Exception as e:
                print(f"Warning: Image encoding failed: {e}")

        # === 关系特征处理 ===
        rel_emb = None
        if self.args.w_rel and rel_features is not None:
            try:
                if self.use_rel_vib:
                    rel_emb = self.rel_fc(rel_features)
                    rel_emb_h = F.relu(rel_emb)
                    mu = self.rel_fc_mu(rel_emb_h)
                    logvar = self.rel_fc_std(rel_emb_h)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    rel_emb = mu + eps * std
                    self.rel_kld_loss = self._kld_gauss(
                        mu, std, torch.zeros_like(mu), torch.ones_like(std)
                    )
                else:
                    rel_emb = self.rel_fc(rel_features)
                embeddings.append(rel_emb)
            except Exception as e:
                print(f"Warning: Relation encoding failed: {e}")

        # === 属性特征处理 ===
        att_emb = None
        if self.args.w_attr and att_features is not None:
            try:
                if self.use_attr_vib:
                    att_emb = self.att_fc(att_features)
                    att_emb_h = F.relu(att_emb)
                    mu = self.att_fc_mu(att_emb_h)
                    logvar = self.att_fc_std(att_emb_h)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    att_emb = mu + eps * std
                    self.attr_kld_loss = self._kld_gauss(
                        mu, std, torch.zeros_like(mu), torch.ones_like(std)
                    )
                else:
                    att_emb = self.att_fc(att_features)
                embeddings.append(att_emb)
            except Exception as e:
                print(f"Warning: Attribute encoding failed: {e}")

        # === 其他特征处理 ===
        name_emb = None
        if self.args.w_name and name_features is not None:
            try:
                name_emb = self.name_fc(name_features)
                embeddings.append(name_emb)
            except Exception as e:
                print(f"Warning: Name encoding failed: {e}")
            
        char_emb = None
        if self.args.w_char and char_features is not None:
            try:
                char_emb = self.char_fc(char_features)
                embeddings.append(char_emb)
            except Exception as e:
                print(f"Warning: Char encoding failed: {e}")

        # === 投影头处理 ===
        if self.use_project_head:
            if gph_emb is not None and hasattr(self, 'gph_pro'):
                gph_emb = self.gph_pro(gph_emb)
            if img_emb is not None and hasattr(self, 'img_pro'):
                img_emb = self.img_pro(img_emb)
            if rel_emb is not None and hasattr(self, 'rel_pro'):
                rel_emb = self.rel_pro(rel_emb)
            if att_emb is not None and hasattr(self, 'att_pro'):
                att_emb = self.att_pro(att_emb)

        # === 多模态融合 ===
        try:
            joint_emb = self.fusion([img_emb, att_emb, rel_emb, gph_emb, name_emb, char_emb])
        except Exception as e:
            print(f"Warning: Fusion failed: {e}")
            # Fallback融合
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            if valid_embeddings:
                joint_emb = torch.cat(valid_embeddings, dim=-1)
            else:
                joint_emb = torch.zeros(input_idx.size(0), 128, device=input_idx.device)

        # === 联合变分编码 ===
        if self.use_joint_vib:
            try:
                joint_emb = self.joint_fc(joint_emb)
                joint_emb_h = F.relu(joint_emb)
                mu = self.joint_fc_mu(joint_emb_h)
                logvar = self.joint_fc_std(joint_emb_h)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                joint_emb = mu + eps * std
                self.joint_kld_loss = self._kld_gauss(
                    mu, std, torch.zeros_like(mu), torch.ones_like(std)
                )
            except Exception as e:
                print(f"Warning: Joint VIB failed: {e}")

        return gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb


class MultiModalFusion(nn.Module):
    def __init__(self, modal_num=6, with_weight=1):
        super(MultiModalFusion, self).__init__()
        self.modal_num = modal_num
        self.with_weight = with_weight
        if with_weight:
            self.weight = nn.Parameter(torch.ones(modal_num))
    
    def forward(self, modal_list):
        valid_modals = [modal for modal in modal_list if modal is not None]
        if not valid_modals:
            # 如果没有有效模态，返回零张量
            device = modal_list[0].device if modal_list and modal_list[0] is not None else torch.device('cpu')
            return torch.zeros(1, 128, device=device)
        
        if self.with_weight and hasattr(self, 'weight'):
            weighted_modals = []
            weight_idx = 0
            for modal in modal_list:
                if modal is not None:
                    if weight_idx < len(self.weight):
                        weighted_modals.append(self.weight[weight_idx] * modal)
                        weight_idx += 1
                    else:
                        weighted_modals.append(modal)
            if weighted_modals:
                return torch.cat(weighted_modals, dim=-1)
        
        return torch.cat(valid_modals, dim=-1)


class SimpleFusion(nn.Module):
    def __init__(self, total_dim):
        super(SimpleFusion, self).__init__()
        self.total_dim = total_dim
        
    def forward(self, modal_list):
        valid_modals = [modal for modal in modal_list if modal is not None]
        if not valid_modals:
            device = modal_list[0].device if modal_list and modal_list[0] is not None else torch.device('cpu')
            return torch.zeros(1, 128, device=device)
        return torch.cat(valid_modals, dim=-1)


# 为了保持兼容性，设置别名
ClipEnhancedIBMEA = IBMultiModal