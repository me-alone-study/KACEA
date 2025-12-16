#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的损失函数
主要改进：
1. 跨模态一致性损失
2. 硬负样本挖掘
3. 模态可靠性加权损失
4. 对比学习增强
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
    带硬负样本挖掘的InfoNCE损失
    改进点：更关注难以区分的负样本
    """
    
    def __init__(self, device, temperature=0.05, hard_negative_weight=0.5):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
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
        # 找到每个样本最难的负样本（相似度最高但不是正样本的）
        with torch.no_grad():
            # 创建mask，排除对角线（正样本）
            mask = torch.eye(batch_size, device=self.device).bool()
            sim_masked = sim_matrix.clone()
            sim_masked[mask] = -float('inf')
            
            # 找到最难的负样本
            hard_neg_indices = sim_masked.argmax(dim=1)
        
        # 计算硬负样本的三元组损失
        hard_neg_emb = emb_right[hard_neg_indices]
        pos_sim = (emb_left * emb_right).sum(dim=1)
        neg_sim = (emb_left * hard_neg_emb).sum(dim=1)
        
        # margin-based triplet loss
        margin = 0.3
        triplet_loss = F.relu(neg_sim - pos_sim + margin).mean()
        
        total_loss = standard_loss + self.hard_negative_weight * triplet_loss
        
        return total_loss


class CrossModalConsistencyLoss(nn.Module):
    """
    跨模态一致性损失
    强制不同模态表示同一实体时保持一致
    
    修复：处理不同模态维度不一致的情况
    """
    
    def __init__(self, device, temperature=0.1, consistency_weight=0.5):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.projection_layers = nn.ModuleDict()  # 动态创建投影层
        self.standard_dim = 128  # 标准化维度
    
    def _get_projection(self, dim, name):
        """获取或创建投影层"""
        key = f"{name}_{dim}"
        if key not in self.projection_layers:
            self.projection_layers[key] = nn.Linear(dim, self.standard_dim).to(self.device)
        return self.projection_layers[key]
    
    def forward(self, modal_features, train_links):
        """
        计算跨模态一致性损失
        
        Args:
            modal_features: dict, {modal_name: features [N, D]}
            train_links: 训练对齐链接 [M, 2]
        """
        # 收集有效模态
        valid_modals = {k: v for k, v in modal_features.items() if v is not None}
        
        if len(valid_modals) < 2:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        pair_count = 0
        
        modal_names = list(valid_modals.keys())
        
        # 计算每对模态之间的一致性损失
        for i in range(len(modal_names)):
            for j in range(i + 1, len(modal_names)):
                modal_i = modal_names[i]
                modal_j = modal_names[j]
                
                try:
                    feat_i = valid_modals[modal_i]
                    feat_j = valid_modals[modal_j]
                    
                    # 🔧 修复：投影到相同维度
                    if feat_i.size(-1) != self.standard_dim:
                        proj_i = self._get_projection(feat_i.size(-1), modal_i)
                        feat_i = proj_i(feat_i)
                    if feat_j.size(-1) != self.standard_dim:
                        proj_j = self._get_projection(feat_j.size(-1), modal_j)
                        feat_j = proj_j(feat_j)
                    
                    feat_i = F.normalize(feat_i, dim=-1)
                    feat_j = F.normalize(feat_j, dim=-1)
                    
                    # 确保batch size一致
                    min_size = min(feat_i.size(0), feat_j.size(0))
                    feat_i = feat_i[:min_size]
                    feat_j = feat_j[:min_size]
                    
                    # 过滤有效的训练链接
                    valid_links = train_links[train_links[:, 0] < min_size]
                    valid_links = valid_links[valid_links[:, 1] < min_size]
                    
                    if len(valid_links) == 0:
                        continue
                    
                    # 获取对齐实体的特征
                    aligned_i = feat_i[valid_links[:, 0]]
                    aligned_j = feat_j[valid_links[:, 1]]
                    
                    # 计算对比损失（现在维度一致了）
                    sim_matrix = torch.mm(aligned_i, aligned_j.t()) / self.temperature
                    batch_size = sim_matrix.size(0)
                    labels = torch.arange(batch_size, device=self.device)
                    
                    loss_ij = F.cross_entropy(sim_matrix, labels)
                    loss_ji = F.cross_entropy(sim_matrix.t(), labels)
                    
                    total_loss += (loss_ij + loss_ji) / 2
                    pair_count += 1
                    
                except Exception as e:
                    # 跳过出错的模态对
                    continue
        
        if pair_count > 0:
            return self.consistency_weight * total_loss / pair_count
        else:
            return torch.tensor(0.0, device=self.device)


class ModalityReliabilityWeightedLoss(nn.Module):
    """
    基于模态可靠性的加权损失
    对于低质量/缺失模态，降低其在总损失中的权重
    """
    
    def __init__(self, device, base_temperature=0.05):
        super().__init__()
        self.device = device
        self.base_temperature = base_temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def compute_reliability_scores(self, features, reference_features=None):
        """
        计算每个样本的可靠性分数
        基于特征方差和与参考特征的一致性
        """
        if features is None:
            return None
        
        # 计算特征方差（低方差可能表示随机填充）
        feature_var = features.var(dim=-1)
        var_score = torch.sigmoid(feature_var * 10 - 0.5)
        
        # 计算特征范数
        feature_norm = features.norm(dim=-1)
        norm_score = torch.sigmoid(feature_norm - 0.5)
        
        # 如果有参考特征，计算一致性
        if reference_features is not None:
            ref_norm = F.normalize(reference_features, dim=-1)
            feat_norm = F.normalize(features, dim=-1)
            consistency = (ref_norm * feat_norm).sum(dim=-1)
            consistency_score = (consistency + 1) / 2  # 归一化到[0, 1]
            reliability = (var_score + norm_score + consistency_score) / 3
        else:
            reliability = (var_score + norm_score) / 2
        
        return reliability
    
    def forward(self, emb, train_links, reliability_scores=None):
        """
        加权InfoNCE损失
        """
        emb = F.normalize(emb)
        emb_left = emb[train_links[:, 0]]
        emb_right = emb[train_links[:, 1]]
        
        batch_size = emb_left.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(emb_left, emb_right.t()) / self.base_temperature
        
        labels = torch.arange(batch_size, device=self.device)
        
        # 计算每个样本的损失
        loss_per_sample = self.ce_loss(sim_matrix, labels)
        
        # 如果有可靠性分数，进行加权
        if reliability_scores is not None:
            # 获取训练样本的可靠性分数
            rel_left = reliability_scores[train_links[:, 0]]
            rel_right = reliability_scores[train_links[:, 1]]
            sample_weights = (rel_left + rel_right) / 2
            
            # 归一化权重
            sample_weights = sample_weights / (sample_weights.sum() + 1e-8) * batch_size
            
            weighted_loss = (loss_per_sample * sample_weights).mean()
        else:
            weighted_loss = loss_per_sample.mean()
        
        return weighted_loss


class CLIPAlignmentLoss(nn.Module):
    """修正的CLIP对齐损失函数"""
    
    def __init__(self, device, temperature=0.07):
        super(CLIPAlignmentLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, clip_features, train_links):
        """
        计算CLIP风格的对比学习损失
        """
        if not isinstance(clip_features, dict):
            return torch.tensor(0.0, device=self.device)
            
        if 'image_embeds' not in clip_features or 'text_embeds' not in clip_features:
            return torch.tensor(0.0, device=self.device)
        
        img_embeds = clip_features['image_embeds']
        text_embeds = clip_features['text_embeds']
        
        if img_embeds is None or text_embeds is None:
            return torch.tensor(0.0, device=self.device)
        
        img_embeds = img_embeds.to(self.device)
        text_embeds = text_embeds.to(self.device)
        
        img_embeds = F.normalize(img_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        min_batch_size = min(img_embeds.size(0), text_embeds.size(0))
        img_embeds = img_embeds[:min_batch_size]
        text_embeds = text_embeds[:min_batch_size]
        
        batch_size = img_embeds.size(0)
        
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device)
        
        sim_i2t = torch.matmul(img_embeds, text_embeds.T) / self.temperature
        sim_t2i = torch.matmul(text_embeds, img_embeds.T) / self.temperature
        
        labels = torch.arange(batch_size, device=self.device)
        
        loss_i2t = self.ce_loss(sim_i2t, labels)
        loss_t2i = self.ce_loss(sim_t2i, labels)
        
        return (loss_i2t + loss_t2i) / 2


class CrossModalAlignmentLoss(nn.Module):
    """跨模态对齐损失，用于entity alignment任务"""
    
    def __init__(self, device, temperature=0.07, margin=0.1):
        super(CrossModalAlignmentLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, embeddings, train_links, modal_features=None):
        """
        跨模态对齐损失
        """
        if embeddings is None:
            return torch.tensor(0.0, device=self.device)
        
        try:
            total_loss = 0.0
            loss_count = 0
            
            # 1. 主要的实体对齐损失
            embeddings = F.normalize(embeddings, dim=-1)
            left_embeds = embeddings[train_links[:, 0]]
            right_embeds = embeddings[train_links[:, 1]]
            
            sim_matrix = torch.matmul(left_embeds, right_embeds.T) / self.temperature
            batch_size = sim_matrix.size(0)
            labels = torch.arange(batch_size, device=self.device)
            
            loss_l2r = self.ce_loss(sim_matrix, labels)
            loss_r2l = self.ce_loss(sim_matrix.T, labels)
            
            alignment_loss = (loss_l2r + loss_r2l) / 2
            total_loss += alignment_loss
            loss_count += 1
            
            # 2. 跨模态一致性损失
            if modal_features is not None and isinstance(modal_features, dict):
                modal_consistency_loss = 0.0
                modal_count = 0
                
                valid_modals = {k: v for k, v in modal_features.items() 
                              if v is not None and k != 'clip'}
                
                modal_list = list(valid_modals.values())
                
                for i in range(len(modal_list)):
                    for j in range(i + 1, len(modal_list)):
                        try:
                            feat_i = F.normalize(modal_list[i], dim=-1)
                            feat_j = F.normalize(modal_list[j], dim=-1)
                            
                            min_size = min(feat_i.size(0), feat_j.size(0))
                            feat_i = feat_i[:min_size]
                            feat_j = feat_j[:min_size]
                            
                            valid_train_links = train_links[train_links[:, 0] < min_size]
                            valid_train_links = valid_train_links[valid_train_links[:, 1] < min_size]
                            
                            if len(valid_train_links) == 0:
                                continue
                            
                            feat_i_aligned = feat_i[valid_train_links[:, 0]]
                            feat_j_aligned = feat_j[valid_train_links[:, 1]]
                            
                            cosine_sim = F.cosine_similarity(feat_i_aligned, feat_j_aligned, dim=-1)
                            consistency_loss = 1.0 - cosine_sim.mean()
                            
                            modal_consistency_loss += consistency_loss
                            modal_count += 1
                            
                        except Exception as e:
                            continue
                
                if modal_count > 0:
                    total_loss += 0.1 * modal_consistency_loss / modal_count
            
            return total_loss
            
        except Exception as e:
            return torch.tensor(0.0, device=self.device)


class CLIPAwareContrastiveLoss(nn.Module):
    """CLIP感知的对比学习损失"""
    
    def __init__(self, device, temperature=0.07, clip_weight=0.1, entity_weight=1.0):
        super(CLIPAwareContrastiveLoss, self).__init__()
        self.device = device
        self.clip_loss = CLIPAlignmentLoss(device, temperature)
        self.alignment_loss = CrossModalAlignmentLoss(device, temperature)
        self.clip_weight = clip_weight
        self.entity_weight = entity_weight
    
    def forward(self, joint_embeddings, train_links, clip_features=None, modal_features=None):
        """
        计算综合损失
        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        try:
            if joint_embeddings is not None:
                entity_loss = self.alignment_loss(joint_embeddings, train_links, modal_features)
                total_loss += self.entity_weight * entity_loss
            
            if clip_features is not None and self.clip_weight > 0:
                clip_loss = self.clip_loss(clip_features, train_links)
                total_loss += self.clip_weight * clip_loss
            
            return total_loss
            
        except Exception as e:
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


class ImprovedComprehensiveLoss(nn.Module):
    """
    改进的综合损失函数
    主要改进：
    1. 硬负样本挖掘
    2. 跨模态一致性损失
    3. 模态可靠性加权
    """
    
    def __init__(self, args, device):
        super(ImprovedComprehensiveLoss, self).__init__()
        self.args = args
        self.device = device
        
        # 基础损失函数 - 使用改进版本
        self.hard_nce = HardNegativeInfoNCE(
            device, 
            temperature=args.tau,
            hard_negative_weight=getattr(args, 'hard_negative_weight', 0.5)
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
            num_losses += 2  # joint loss + cross modal loss
            
            self.adaptive_weighting = AdaptiveLossWeighting(num_losses, device)
    
    def forward(self, embeddings_dict, train_links, model):
        """
        计算综合损失
        """
        try:
            total_loss = 0.0
            loss_components = {}
            individual_losses = []
            
            # 1. 各模态的损失（使用硬负样本挖掘）
            if self.args.w_gcn and 'graph' in embeddings_dict and embeddings_dict['graph'] is not None:
                gph_loss = self.hard_nce(embeddings_dict['graph'], train_links)
                if hasattr(model, 'kld_loss') and model.kld_loss > 0:
                    gph_loss += self.args.Beta_g * model.kld_loss
                loss_components['graph'] = gph_loss
                individual_losses.append(gph_loss)
            
            if self.args.w_img and 'image' in embeddings_dict and embeddings_dict['image'] is not None:
                img_loss = self.hard_nce(embeddings_dict['image'], train_links)
                if hasattr(model, 'img_kld_loss') and model.img_kld_loss > 0:
                    img_loss += self.args.Beta_i * model.img_kld_loss
                loss_components['image'] = img_loss
                individual_losses.append(img_loss)
            
            if self.args.w_rel and 'relation' in embeddings_dict and embeddings_dict['relation'] is not None:
                rel_loss = self.hard_nce(embeddings_dict['relation'], train_links)
                if hasattr(model, 'rel_kld_loss') and model.rel_kld_loss > 0:
                    rel_loss += self.args.Beta_r * model.rel_kld_loss
                loss_components['relation'] = rel_loss
                individual_losses.append(rel_loss)
            
            if self.args.w_attr and 'attribute' in embeddings_dict and embeddings_dict['attribute'] is not None:
                attr_loss = self.hard_nce(embeddings_dict['attribute'], train_links)
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
                    joint_loss = self.hard_nce(embeddings_dict['joint'], train_links)
                    if hasattr(model, 'joint_kld_loss') and model.joint_kld_loss > 0:
                        joint_loss += getattr(self.args, 'joint_beta', 1.0) * model.joint_kld_loss
                else:
                    joint_loss = self.ms_loss(embeddings_dict['joint'], train_links)
                    joint_loss *= getattr(self.args, 'joint_beta', 1.0)
                
                loss_components['joint'] = joint_loss
                individual_losses.append(joint_loss)
            
            # 4. 跨模态一致性损失（新增！）
            if hasattr(model, 'modal_features') and model.modal_features:
                cross_modal_loss = self.cross_modal_loss(model.modal_features, train_links)
                loss_components['cross_modal'] = cross_modal_loss
                individual_losses.append(cross_modal_loss)
            
            # 5. 跨模态对齐损失（从模型获取）
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