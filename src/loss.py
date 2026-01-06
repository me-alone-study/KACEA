#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的损失函数 - 增强版
主要改进：
1. 跨模态一致性损失
2. 硬负样本挖掘 - [改进] 增大margin
3. 模态可靠性加权损失
4. 对比学习增强
5. [新增] 置信度加权损失
"""

import torch
import torch.nn.functional as F
from torch import nn
from pytorch_metric_learning import losses, miners

try:
    from models import *
    from utils import *
except:
    from src.models import *
    from src.utils import *


class MsLoss(nn.Module):
    def __init__(self, device, thresh=0.5, scale_pos=0.1, scale_neg=40.0):
        super(MsLoss, self).__init__()
        self.device = device
        alpha, beta, base = scale_pos, scale_neg, thresh
        self.loss_func = losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)

    def sim(self, emb_left, emb_right):
        return emb_left.mm(emb_right.t())

    def forward(self, emb, train_links):
        emb = F.normalize(emb)
        emb_train_left = emb[train_links[:, 0]]
        emb_train_right = emb[train_links[:, 1]]
        labels = torch.arange(emb_train_left.size(0))
        embeddings = torch.cat([emb_train_left, emb_train_right], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        loss = self.loss_func(embeddings, labels)
        return loss


class InfoNCE_loss(nn.Module):
    def __init__(self, device, temperature=0.05) -> None:
        super().__init__()
        self.device = device
        self.t = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def sim(self, emb_left, emb_right):
        return emb_left.mm(emb_right.t())

    def forward(self, emb, train_links):
        emb = F.normalize(emb)
        emb_train_left = emb[train_links[:, 0]]
        emb_train_right = emb[train_links[:, 1]]

        score = self.sim(emb_train_left, emb_train_right)
        bsize = emb_train_left.size()[0]
        label = torch.arange(bsize, dtype=torch.long).to(self.device)

        loss = self.ce_loss(score / self.t, label)
        return loss


class HardNegativeInfoNCE(nn.Module):
    """
    带硬负样本挖掘的InfoNCE损失 - 改进版
    [改进] 增大margin从0.3到0.5，增强负样本区分能力
    """
    
    def __init__(self, device, temperature=0.05, hard_negative_weight=0.5, margin=0.5):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.margin = margin  # [改进] 从0.3增加到0.5
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, emb, train_links):
        emb = F.normalize(emb)
        emb_left = emb[train_links[:, 0]]
        emb_right = emb[train_links[:, 1]]
        
        batch_size = emb_left.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(emb_left, emb_right.t()) / self.temperature
        
        # 标准InfoNCE损失
        labels = torch.arange(batch_size, device=self.device)
        loss_l2r = self.ce_loss(sim_matrix, labels)
        loss_r2l = self.ce_loss(sim_matrix.t(), labels)
        standard_loss = (loss_l2r + loss_r2l) / 2
        
        # 硬负样本损失
        with torch.no_grad():
            mask = torch.eye(batch_size, device=self.device).bool()
            sim_masked = sim_matrix.clone()
            sim_masked[mask] = -float('inf')
            hard_neg_indices = sim_masked.argmax(dim=1)
        
        hard_neg_emb = emb_right[hard_neg_indices]
        pos_sim = (emb_left * emb_right).sum(dim=1)
        neg_sim = (emb_left * hard_neg_emb).sum(dim=1)
        
        # [改进] 使用更大的margin
        triplet_loss = F.relu(neg_sim - pos_sim + self.margin).mean()
        
        total_loss = standard_loss + self.hard_negative_weight * triplet_loss
        
        return total_loss


class CrossModalConsistencyLoss(nn.Module):
    """
    跨模态一致性损失
    强制不同模态表示同一实体时保持一致
    """
    
    def __init__(self, device, temperature=0.1, consistency_weight=0.5):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.projection_layers = nn.ModuleDict()
        self.standard_dim = 128
    
    def _get_projection(self, dim, name):
        """获取或创建投影层"""
        key = f"{name}_{dim}"
        if key not in self.projection_layers:
            self.projection_layers[key] = nn.Linear(dim, self.standard_dim).to(self.device)
        return self.projection_layers[key]
    
    def forward(self, modal_features, train_links):
        """计算跨模态一致性损失"""
        valid_modals = {k: v for k, v in modal_features.items() if v is not None}
        
        if len(valid_modals) < 2:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        pair_count = 0
        
        modal_names = list(valid_modals.keys())
        
        for i in range(len(modal_names)):
            for j in range(i + 1, len(modal_names)):
                modal_i = modal_names[i]
                modal_j = modal_names[j]
                
                try:
                    feat_i = valid_modals[modal_i]
                    feat_j = valid_modals[modal_j]
                    
                    if feat_i.size(-1) != self.standard_dim:
                        proj_i = self._get_projection(feat_i.size(-1), modal_i)
                        feat_i = proj_i(feat_i)
                    if feat_j.size(-1) != self.standard_dim:
                        proj_j = self._get_projection(feat_j.size(-1), modal_j)
                        feat_j = proj_j(feat_j)
                    
                    feat_i = F.normalize(feat_i, dim=-1)
                    feat_j = F.normalize(feat_j, dim=-1)
                    
                    min_size = min(feat_i.size(0), feat_j.size(0))
                    feat_i = feat_i[:min_size]
                    feat_j = feat_j[:min_size]
                    
                    valid_links = train_links[train_links[:, 0] < min_size]
                    valid_links = valid_links[valid_links[:, 1] < min_size]
                    
                    if len(valid_links) == 0:
                        continue
                    
                    aligned_i = feat_i[valid_links[:, 0]]
                    aligned_j = feat_j[valid_links[:, 1]]
                    
                    sim_matrix = torch.mm(aligned_i, aligned_j.t()) / self.temperature
                    batch_size = sim_matrix.size(0)
                    labels = torch.arange(batch_size, device=self.device)
                    
                    loss_ij = F.cross_entropy(sim_matrix, labels)
                    loss_ji = F.cross_entropy(sim_matrix.t(), labels)
                    
                    total_loss += (loss_ij + loss_ji) / 2
                    pair_count += 1
                    
                except Exception as e:
                    continue
        
        if pair_count > 0:
            return self.consistency_weight * (total_loss / pair_count)
        else:
            return torch.tensor(0.0, device=self.device)


class ModalityReliabilityWeightedLoss(nn.Module):
    """
    模态可靠性加权损失 - 改进版
    根据模态的可靠性动态调整损失权重
    [新增] 对于低质量模态自动降低权重
    """
    
    def __init__(self, device, base_temperature=0.1, min_weight=0.1):
        super().__init__()
        self.device = device
        self.base_temperature = base_temperature
        self.min_weight = min_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def compute_reliability(self, features):
        """
        计算特征可靠性
        基于特征方差和范数
        """
        if features is None:
            return torch.zeros(1, device=self.device)
        
        # 检查非零比例
        non_zero_mask = features.abs().sum(dim=-1) > 1e-6
        non_zero_ratio = non_zero_mask.float().mean()
        
        # 方差评分
        var = features.var(dim=-1)
        var_score = torch.sigmoid(var.mean() * 10 - 0.5)
        
        # 范数评分
        norm = features.norm(dim=-1)
        norm_score = torch.sigmoid(norm.mean() - 1.0)
        
        reliability = (non_zero_ratio + var_score + norm_score) / 3.0
        return reliability.clamp(min=self.min_weight)
    
    def forward(self, embeddings_dict, train_links, modal_weights=None):
        """
        计算加权损失
        """
        total_loss = 0.0
        total_weight = 0.0
        
        for modal_name, emb in embeddings_dict.items():
            if emb is None:
                continue
            
            try:
                emb = F.normalize(emb)
                emb_left = emb[train_links[:, 0]]
                emb_right = emb[train_links[:, 1]]
                
                sim_matrix = torch.mm(emb_left, emb_right.t()) / self.base_temperature
                batch_size = emb_left.size(0)
                labels = torch.arange(batch_size, device=self.device)
                
                loss = self.ce_loss(sim_matrix, labels).mean()
                
                # 计算或使用提供的权重
                if modal_weights is not None and modal_name in modal_weights:
                    weight = modal_weights[modal_name]
                else:
                    weight = self.compute_reliability(emb)
                
                total_loss += weight * loss
                total_weight += weight
                
            except Exception as e:
                continue
        
        if total_weight > 0:
            return total_loss / total_weight
        else:
            return torch.tensor(0.0, device=self.device)


class CLIPAwareContrastiveLoss(nn.Module):
    """
    CLIP感知的对比学习损失
    利用CLIP的跨模态对齐能力增强对齐质量
    """
    
    def __init__(self, device, temperature=0.07, clip_weight=0.1, entity_weight=1.0):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.clip_weight = clip_weight
        self.entity_weight = entity_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, entity_emb, train_links, clip_features=None, modal_features=None):
        """
        计算CLIP增强的对比损失
        """
        total_loss = 0.0
        loss_count = 0
        
        # 1. 实体对齐损失
        if entity_emb is not None:
            try:
                emb = F.normalize(entity_emb)
                emb_left = emb[train_links[:, 0]]
                emb_right = emb[train_links[:, 1]]
                
                sim_matrix = torch.mm(emb_left, emb_right.t()) / self.temperature
                batch_size = emb_left.size(0)
                labels = torch.arange(batch_size, device=self.device)
                
                entity_loss = self.ce_loss(sim_matrix, labels)
                total_loss += self.entity_weight * entity_loss
                loss_count += 1
            except Exception as e:
                pass
        
        # 2. CLIP图文对齐损失
        if clip_features is not None and 'image_embeds' in clip_features and 'text_embeds' in clip_features:
            try:
                img_emb = clip_features['image_embeds']
                txt_emb = clip_features['text_embeds']
                
                # 确保在同一设备
                if img_emb.device != self.device:
                    img_emb = img_emb.to(self.device)
                if txt_emb.device != self.device:
                    txt_emb = txt_emb.to(self.device)
                
                # 获取对齐实体的特征
                min_size = min(img_emb.size(0), txt_emb.size(0))
                valid_links = train_links[train_links[:, 0] < min_size]
                valid_links = valid_links[valid_links[:, 1] < min_size]
                
                if len(valid_links) > 0:
                    img_aligned = img_emb[valid_links[:, 0]]
                    txt_aligned = txt_emb[valid_links[:, 1]]
                    
                    img_aligned = F.normalize(img_aligned, dim=-1)
                    txt_aligned = F.normalize(txt_aligned, dim=-1)
                    
                    clip_sim = torch.mm(img_aligned, txt_aligned.t()) / self.temperature
                    batch_size = clip_sim.size(0)
                    labels = torch.arange(batch_size, device=self.device)
                    
                    clip_loss = (self.ce_loss(clip_sim, labels) + 
                               self.ce_loss(clip_sim.t(), labels)) / 2
                    total_loss += self.clip_weight * clip_loss
                    loss_count += 1
            except Exception as e:
                pass
        
        if loss_count > 0:
            return total_loss / loss_count
        else:
            return torch.tensor(0.0, device=self.device)


class VIBLoss(nn.Module):
    """变分信息瓶颈损失"""
    
    def __init__(self, device, beta=0.001):
        super(VIBLoss, self).__init__()
        self.device = device
        self.beta = beta
    
    def forward(self, kld_losses):
        """计算VIB损失"""
        try:
            total_kld = 0.0
            count = 0
            
            for modal_name, kld_loss in kld_losses.items():
                if kld_loss > 0:
                    total_kld += kld_loss
                    count += 1
            
            if count > 0:
                return self.beta * (total_kld / count)
            else:
                return torch.tensor(0.0, device=self.device)
                
        except Exception as e:
            return torch.tensor(0.0, device=self.device)


class AdaptiveLossWeighting(nn.Module):
    """自适应损失权重调整"""
    
    def __init__(self, num_losses, device):
        super(AdaptiveLossWeighting, self).__init__()
        self.device = device
        self.num_losses = num_losses
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses):
        """计算加权损失"""
        try:
            if len(losses) != self.num_losses:
                return sum(losses) / max(len(losses), 1)
            
            weighted_losses = []
            for i, loss in enumerate(losses):
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss + self.log_vars[i]
                weighted_losses.append(weighted_loss)
            
            return sum(weighted_losses)
            
        except Exception as e:
            return sum(losses) / max(len(losses), 1)


class ConfidenceWeightedLoss(nn.Module):
    """
    [新增] 置信度加权损失
    根据预测置信度调整损失权重，降低不确定样本的影响
    """
    
    def __init__(self, device, temperature=0.05, confidence_threshold=0.5):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
    
    def forward(self, emb, train_links):
        emb = F.normalize(emb)
        emb_left = emb[train_links[:, 0]]
        emb_right = emb[train_links[:, 1]]
        
        batch_size = emb_left.size(0)
        
        sim_matrix = torch.mm(emb_left, emb_right.t()) / self.temperature
        
        # 计算置信度（softmax后的最大值）
        probs = F.softmax(sim_matrix, dim=1)
        confidence = probs.max(dim=1)[0]
        
        # 置信度加权
        weights = torch.where(
            confidence > self.confidence_threshold,
            torch.ones_like(confidence),
            confidence / self.confidence_threshold
        )
        
        # 计算加权损失
        labels = torch.arange(batch_size, device=self.device)
        per_sample_loss = F.cross_entropy(sim_matrix, labels, reduction='none')
        weighted_loss = (per_sample_loss * weights).mean()
        
        return weighted_loss


class ImprovedComprehensiveLoss(nn.Module):
    """
    改进的综合损失函数 - 增强版
    主要改进：
    1. 硬负样本挖掘 - [改进] 更大的margin
    2. 跨模态一致性损失
    3. 模态可靠性加权
    4. [新增] 置信度加权选项
    """
    
    def __init__(self, args, device):
        super(ImprovedComprehensiveLoss, self).__init__()
        self.args = args
        self.device = device
        
        # [改进] 增大margin
        self.hard_nce = HardNegativeInfoNCE(
            device, 
            temperature=args.tau,
            hard_negative_weight=getattr(args, 'hard_negative_weight', 0.5),
            margin=getattr(args, 'triplet_margin', 0.5)  # [改进] 默认0.5
        )
        self.ms_loss = MsLoss(
            device, 
            thresh=getattr(args, 'ms_base', 0.5),
            scale_pos=getattr(args, 'ms_alpha', 0.1), 
            scale_neg=getattr(args, 'ms_beta', 40.0)
        )
        
        # 跨模态一致性损失
        self.cross_modal_loss = CrossModalConsistencyLoss(
            device,
            temperature=0.1,
            consistency_weight=getattr(args, 'cross_modal_weight', 0.3)
        )
        
        # 模态可靠性加权损失
        self.reliability_weighted_loss = ModalityReliabilityWeightedLoss(
            device,
            base_temperature=args.tau
        )
        
        # [新增] 置信度加权损失（可选）
        self.use_confidence_weighting = getattr(args, 'use_confidence_weighting', False)
        if self.use_confidence_weighting:
            self.confidence_loss = ConfidenceWeightedLoss(
                device, 
                temperature=args.tau,
                confidence_threshold=getattr(args, 'confidence_threshold', 0.5)
            )
        
        # CLIP相关损失
        if getattr(args, 'use_clip', False):
            self.clip_loss = CLIPAwareContrastiveLoss(
                device=device,
                temperature=getattr(args, 'clip_temperature', 0.07),
                clip_weight=getattr(args, 'clip_weight', 0.1),
                entity_weight=1.0
            )
        
        # VIB损失
        self.vib_loss = VIBLoss(device, beta=0.001)
        
        # 自适应权重
        self.use_adaptive_weighting = getattr(args, 'use_adaptive_weighting', False)
        if self.use_adaptive_weighting:
            num_losses = 0
            if args.w_gcn: num_losses += 1
            if args.w_img: num_losses += 1
            if args.w_rel: num_losses += 1
            if args.w_attr: num_losses += 1
            if getattr(args, 'use_clip', False): num_losses += 1
            num_losses += 2
            
            self.adaptive_weighting = AdaptiveLossWeighting(num_losses, device)
    
    def forward(self, embeddings_dict, train_links, model):
        """计算综合损失"""
        try:
            total_loss = 0.0
            loss_components = {}
            individual_losses = []
            
            # 选择基础损失函数
            base_loss_fn = self.confidence_loss if self.use_confidence_weighting else self.hard_nce
            
            # 1. 各模态的损失
            if self.args.w_gcn and 'graph' in embeddings_dict and embeddings_dict['graph'] is not None:
                gph_loss = base_loss_fn(embeddings_dict['graph'], train_links)
                if hasattr(model, 'kld_loss') and model.kld_loss > 0:
                    gph_loss += self.args.Beta_g * model.kld_loss
                loss_components['graph'] = gph_loss
                individual_losses.append(gph_loss)
            
            if self.args.w_img and 'image' in embeddings_dict and embeddings_dict['image'] is not None:
                img_loss = base_loss_fn(embeddings_dict['image'], train_links)
                if hasattr(model, 'img_kld_loss') and model.img_kld_loss > 0:
                    img_loss += self.args.Beta_i * model.img_kld_loss
                loss_components['image'] = img_loss
                individual_losses.append(img_loss)
            
            if self.args.w_rel and 'relation' in embeddings_dict and embeddings_dict['relation'] is not None:
                rel_loss = base_loss_fn(embeddings_dict['relation'], train_links)
                if hasattr(model, 'rel_kld_loss') and model.rel_kld_loss > 0:
                    rel_loss += self.args.Beta_r * model.rel_kld_loss
                loss_components['relation'] = rel_loss
                individual_losses.append(rel_loss)
            
            if self.args.w_attr and 'attribute' in embeddings_dict and embeddings_dict['attribute'] is not None:
                attr_loss = base_loss_fn(embeddings_dict['attribute'], train_links)
                if hasattr(model, 'attr_kld_loss') and model.attr_kld_loss > 0:
                    attr_loss += self.args.Beta_a * model.attr_kld_loss
                loss_components['attribute'] = attr_loss
                individual_losses.append(attr_loss)
            
            # 2. CLIP损失
            if getattr(self.args, 'use_clip', False) and hasattr(model, 'clip_features'):
                clip_loss = self.clip_loss(
                    embeddings_dict.get('joint', None),
                    train_links,
                    model.clip_features,
                    embeddings_dict
                )
                loss_components['clip'] = clip_loss
                individual_losses.append(clip_loss)
            
            # 3. 联合损失
            if 'joint' in embeddings_dict and embeddings_dict['joint'] is not None:
                if getattr(self.args, 'use_joint_vib', False):
                    joint_loss = base_loss_fn(embeddings_dict['joint'], train_links)
                    if hasattr(model, 'joint_kld_loss') and model.joint_kld_loss > 0:
                        joint_loss += getattr(self.args, 'joint_beta', 1.0) * model.joint_kld_loss
                else:
                    joint_loss = self.ms_loss(embeddings_dict['joint'], train_links)
                    joint_loss *= getattr(self.args, 'joint_beta', 1.0)
                
                loss_components['joint'] = joint_loss
                individual_losses.append(joint_loss)
            
            # 4. 跨模态一致性损失
            if hasattr(model, 'modal_features') and model.modal_features:
                cross_modal_loss = self.cross_modal_loss(model.modal_features, train_links)
                loss_components['cross_modal'] = cross_modal_loss
                individual_losses.append(cross_modal_loss)
            
            # 5. 跨模态对齐损失
            if hasattr(model, 'get_cross_modal_alignment_loss'):
                alignment_loss = model.get_cross_modal_alignment_loss(train_links)
                if alignment_loss > 0:
                    loss_components['alignment'] = alignment_loss
                    individual_losses.append(0.2 * alignment_loss)
            
            # 6. 组合损失
            if self.use_adaptive_weighting and len(individual_losses) > 1:
                total_loss = self.adaptive_weighting(individual_losses)
            else:
                total_loss = sum(individual_losses) if individual_losses else torch.tensor(0.0, device=self.device)
            
            return total_loss, loss_components
            
        except Exception as e:
            print(f"Loss computation error: {e}")
            return torch.tensor(0.0, device=self.device), {}


# 保持向后兼容
ComprehensiveLoss = ImprovedComprehensiveLoss
