#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


class CLIPAlignmentLoss(nn.Module):
    """修正后的CLIP对齐损失函数"""
    
    def __init__(self, device, temperature=0.07):
        super(CLIPAlignmentLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, clip_features, train_links):
        """
        计算CLIP风格的对比学习损失
        
        Args:
            clip_features: dict包含'image_embeds'和'text_embeds'
            train_links: 训练链接 [M, 2]
        """
        if 'image_embeds' not in clip_features or 'text_embeds' not in clip_features:
            return torch.tensor(0.0, device=self.device)
        
        img_embeds = clip_features['image_embeds']
        text_embeds = clip_features['text_embeds']
        
        # 归一化特征
        img_embeds = F.normalize(img_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # 对于同一实体，图像特征和文本特征应该相似
        # 这里我们假设img_embeds[i]和text_embeds[i]对应同一实体
        batch_size = img_embeds.size(0)
        
        # 计算图像到文本的相似度矩阵
        sim_i2t = torch.matmul(img_embeds, text_embeds.T) / self.temperature
        sim_t2i = torch.matmul(text_embeds, img_embeds.T) / self.temperature
        
        # 对角线应该是正样本（同一实体的不同模态）
        labels = torch.arange(batch_size, device=self.device)
        
        # 计算双向对比损失
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
        
        Args:
            embeddings: 融合后的实体嵌入 [N, D]
            train_links: 训练对齐链接 [M, 2]
            modal_features: 各模态特征字典（可选）
        """
        if embeddings is None:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        loss_count = 0
        
        # 1. 主要的实体对齐损失
        embeddings = F.normalize(embeddings, dim=-1)
        left_embeds = embeddings[train_links[:, 0]]   # 左侧KG的实体
        right_embeds = embeddings[train_links[:, 1]]  # 右侧KG的实体
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(left_embeds, right_embeds.T) / self.temperature
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=self.device)
        
        # 双向对比损失
        loss_l2r = self.ce_loss(sim_matrix, labels)
        loss_r2l = self.ce_loss(sim_matrix.T, labels)
        
        alignment_loss = (loss_l2r + loss_r2l) / 2
        total_loss += alignment_loss
        loss_count += 1
        
        # 2. 跨模态一致性损失（如果提供了模态特征）
        if modal_features is not None:
            modal_consistency_loss = 0.0
            modal_count = 0
            
            # 获取有效的模态特征
            valid_modals = {k: v for k, v in modal_features.items() 
                          if v is not None and k != 'clip'}
            
            # 计算不同模态之间的一致性
            modal_list = list(valid_modals.values())
            modal_names = list(valid_modals.keys())
            
            for i in range(len(modal_list)):
                for j in range(i + 1, len(modal_list)):
                    try:
                        # 确保特征维度一致
                        feat_i = F.normalize(modal_list[i], dim=-1)
                        feat_j = F.normalize(modal_list[j], dim=-1)
                        
                        # 对齐实体应该在不同模态中也相似
                        feat_i_aligned = feat_i[train_links[:, 0]]
                        feat_j_aligned = feat_j[train_links[:, 1]]
                        
                        # 计算对齐实体的余弦相似度
                        cosine_sim = F.cosine_similarity(feat_i_aligned, feat_j_aligned, dim=-1)
                        
                        # 鼓励对齐实体在不同模态中相似
                        modal_loss = -torch.log(torch.sigmoid(cosine_sim) + 1e-8).mean()
                        modal_consistency_loss += modal_loss
                        modal_count += 1
                        
                    except Exception as e:
                        # 如果特征维度不匹配，跳过
                        continue
            
            if modal_count > 0:
                modal_consistency_loss /= modal_count
                total_loss += 0.1 * modal_consistency_loss  # 较小的权重
                loss_count += 1
        
        return total_loss / max(loss_count, 1)


class CLIPAwareContrastiveLoss(nn.Module):
    """CLIP感知的对比损失"""
    
    def __init__(self, device, temperature=0.07, clip_weight=0.1, entity_weight=1.0):
        super(CLIPAwareContrastiveLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.clip_weight = clip_weight
        self.entity_weight = entity_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 组件损失
        self.clip_loss = CLIPAlignmentLoss(device, temperature)
        self.alignment_loss = CrossModalAlignmentLoss(device, temperature)
    
    def forward(self, joint_embeddings, train_links, clip_features=None, modal_features=None):
        """
        综合损失函数
        
        Args:
            joint_embeddings: 融合后的实体嵌入 [N, D]
            train_links: 训练对齐链接 [M, 2]
            clip_features: CLIP特征字典 (可选)
            modal_features: 各模态特征字典 (可选)
        """
        total_loss = 0.0
        
        # 1. 主要的实体对齐损失
        if joint_embeddings is not None:
            entity_loss = self.alignment_loss(joint_embeddings, train_links, modal_features)
            total_loss += self.entity_weight * entity_loss
        
        # 2. CLIP对齐损失
        if clip_features is not None and self.clip_weight > 0:
            clip_loss = self.clip_loss(clip_features, train_links)
            total_loss += self.clip_weight * clip_loss
        
        return total_loss


class VIBLoss(nn.Module):
    """变分信息瓶颈损失"""
    
    def __init__(self, device, beta=0.001):
        super(VIBLoss, self).__init__()
        self.device = device
        self.beta = beta
    
    def forward(self, kld_losses):
        """
        计算VIB损失
        
        Args:
            kld_losses: KL散度损失字典
        """
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


class AdaptiveLossWeighting(nn.Module):
    """自适应损失权重调整"""
    
    def __init__(self, num_losses, device):
        super(AdaptiveLossWeighting, self).__init__()
        self.device = device
        self.num_losses = num_losses
        
        # 可学习的权重参数
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses):
        """
        计算加权损失
        
        Args:
            losses: 损失列表
        """
        if len(losses) != self.num_losses:
            # 如果损失数量不匹配，使用简单平均
            return sum(losses) / len(losses)
        
        # 使用不确定性加权
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        
        return sum(weighted_losses)


class EnhancedCLIPLoss(nn.Module):
    """增强的CLIP损失函数"""
    
    def __init__(self, device, temperature=0.07):
        super(EnhancedCLIPLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, clip_features, train_links):
        """
        计算CLIP对比学习损失
        """
        if not clip_features or len(train_links) == 0:
            return torch.tensor(0.0, device=self.device)
        
        try:
            total_loss = 0.0
            loss_count = 0
            
            # 图像-文本对比损失
            if 'image_embeds' in clip_features and 'text_embeds' in clip_features:
                img_embeds = F.normalize(clip_features['image_embeds'], dim=-1)
                text_embeds = F.normalize(clip_features['text_embeds'], dim=-1)
                
                batch_size = min(img_embeds.size(0), text_embeds.size(0))
                if batch_size > 1:
                    sim_i2t = torch.matmul(img_embeds[:batch_size], text_embeds[:batch_size].T) / self.temperature
                    labels = torch.arange(batch_size, device=self.device)
                    loss_i2t = self.ce_loss(sim_i2t, labels)
                    total_loss += loss_i2t
                    loss_count += 1
            
            return total_loss / max(loss_count, 1)
            
        except Exception as e:
            return torch.tensor(0.0, device=self.device)


class KnowledgeAwareLoss(nn.Module):
    """知识感知损失函数"""
    
    def __init__(self, device, structure_weight=0.3, semantic_weight=0.2):
        super(KnowledgeAwareLoss, self).__init__()
        self.device = device
        self.structure_weight = structure_weight
        self.semantic_weight = semantic_weight
        
    def forward(self, embeddings, train_links, graph_adj=None):
        """
        计算知识感知损失
        """
        if embeddings is None or len(train_links) == 0:
            return torch.tensor(0.0, device=self.device)
        
        try:
            total_loss = 0.0
            
            # 结构一致性损失
            if graph_adj is not None and self.structure_weight > 0:
                structure_loss = self._compute_structure_loss(embeddings, train_links, graph_adj)
                total_loss += self.structure_weight * structure_loss
            
            return total_loss
            
        except Exception as e:
            return torch.tensor(0.0, device=self.device)
    
    def _compute_structure_loss(self, embeddings, train_links, graph_adj):
        """计算结构一致性损失"""
        try:
            structure_loss = 0.0
            count = 0
            
            # 只处理部分链接以提高效率
            sample_size = min(len(train_links), 50)
            for i in range(sample_size):
                h, t = train_links[i]
                
                # 简化的结构相似性计算
                h_emb = embeddings[h]
                t_emb = embeddings[t]
                
                # 对齐实体应该相似
                sim = F.cosine_similarity(h_emb, t_emb, dim=0)
                structure_loss += 1 - sim
                count += 1
            
            return structure_loss / max(count, 1)
            
        except Exception as e:
            return torch.tensor(0.0, device=self.device)


class ComprehensiveLoss(nn.Module):
    """综合损失函数，整合所有损失项"""
    
    def __init__(self, args, device):
        super(ComprehensiveLoss, self).__init__()
        self.args = args
        self.device = device
        
        # 基础损失函数
        self.info_nce = InfoNCE_loss(device, temperature=args.tau)
        self.ms_loss = MsLoss(device, 
                             thresh=getattr(args, 'ms_base', 0.5),
                             scale_pos=getattr(args, 'ms_alpha', 0.1), 
                             scale_neg=getattr(args, 'ms_beta', 40.0))
        
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
        
        # 知识感知损失
        self.knowledge_loss = KnowledgeAwareLoss(
            device=device,
            structure_weight=getattr(args, 'structure_weight', 0.3),
            semantic_weight=getattr(args, 'semantic_weight', 0.2)
        )
        
        # 损失权重
        self.use_adaptive_weighting = getattr(args, 'use_adaptive_weighting', False)
        if self.use_adaptive_weighting:
            # 计算可能的损失数量
            num_losses = 0
            if args.w_gcn: num_losses += 1
            if args.w_img: num_losses += 1
            if args.w_rel: num_losses += 1
            if args.w_attr: num_losses += 1
            if getattr(args, 'use_clip', False): num_losses += 1
            num_losses += 1  # joint loss
            
            self.adaptive_weighting = AdaptiveLossWeighting(num_losses, device)
    
    def forward(self, embeddings_dict, train_links, model, graph_adj=None):
        """
        计算综合损失
        
        Args:
            embeddings_dict: 包含各种嵌入的字典
            train_links: 训练链接
            model: 模型实例（用于获取KLD损失和CLIP特征）
            graph_adj: 图邻接矩阵
        """
        total_loss = 0.0
        loss_components = {}
        individual_losses = []
        
        try:
            # 1. 各模态的基础损失
            if self.args.w_gcn and 'graph' in embeddings_dict:
                gph_loss = self.info_nce(embeddings_dict['graph'], train_links)
                if hasattr(model, 'kld_loss') and model.kld_loss > 0:
                    gph_loss += self.args.Beta_g * model.kld_loss
                loss_components['graph'] = gph_loss
                individual_losses.append(gph_loss)
            
            if self.args.w_img and 'image' in embeddings_dict:
                img_loss = self.info_nce(embeddings_dict['image'], train_links)
                if hasattr(model, 'img_kld_loss') and model.img_kld_loss > 0:
                    img_loss += self.args.Beta_i * model.img_kld_loss
                loss_components['image'] = img_loss
                individual_losses.append(img_loss)
            
            if self.args.w_rel and 'relation' in embeddings_dict:
                rel_loss = self.info_nce(embeddings_dict['relation'], train_links)
                if hasattr(model, 'rel_kld_loss') and model.rel_kld_loss > 0:
                    rel_loss += self.args.Beta_r * model.rel_kld_loss
                loss_components['relation'] = rel_loss
                individual_losses.append(rel_loss)
            
            if self.args.w_attr and 'attribute' in embeddings_dict:
                attr_loss = self.info_nce(embeddings_dict['attribute'], train_links)
                if hasattr(model, 'attr_kld_loss') and model.attr_kld_loss > 0:
                    attr_loss += self.args.Beta_a * model.attr_kld_loss
                loss_components['attribute'] = attr_loss
                individual_losses.append(attr_loss)
            
            # 2. CLIP损失
            if getattr(self.args, 'use_clip', False) and hasattr(model, 'clip_features'):
                try:
                    clip_loss = self.clip_loss(
                        embeddings_dict.get('joint', None),
                        train_links,
                        model.clip_features,
                        embeddings_dict
                    )
                    loss_components['clip'] = clip_loss
                    individual_losses.append(clip_loss)
                except:
                    pass
            
            # 3. 联合损失
            if 'joint' in embeddings_dict:
                if getattr(self.args, 'use_joint_vib', False):
                    joint_loss = self.info_nce(embeddings_dict['joint'], train_links)
                    if hasattr(model, 'joint_kld_loss') and model.joint_kld_loss > 0:
                        joint_loss += getattr(self.args, 'joint_beta', 1.0) * model.joint_kld_loss
                else:
                    joint_loss = self.ms_loss(embeddings_dict['joint'], train_links)
                    joint_loss *= getattr(self.args, 'joint_beta', 1.0)
                
                loss_components['joint'] = joint_loss
                individual_losses.append(joint_loss)
            
            # 4. 知识感知损失
            if 'joint' in embeddings_dict:
                try:
                    knowledge_loss = self.knowledge_loss(
                        embeddings_dict['joint'], train_links, graph_adj
                    )
                    loss_components['knowledge'] = knowledge_loss
                    individual_losses.append(knowledge_loss)
                except:
                    pass
            
            # 5. 组合损失
            if self.use_adaptive_weighting and len(individual_losses) > 1:
                total_loss = self.adaptive_weighting(individual_losses)
            else:
                total_loss = sum(individual_losses) if individual_losses else torch.tensor(0.0, device=self.device)
        
        except Exception as e:
            print(f"Warning: Loss computation failed: {e}")
            total_loss = torch.tensor(0.001, device=self.device, requires_grad=True)
            loss_components = {}
        
        return total_loss, loss_components