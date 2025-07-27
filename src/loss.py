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
        label = torch.arange(bsize, dtype=torch.long).cuda(self.device)

        loss = self.ce_loss(score / self.t, label)
        return loss

class CLIPAlignmentLoss(nn.Module):
    """CLIP对齐损失函数"""
    
    def __init__(self, device, temperature=0.07):
        super(CLIPAlignmentLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, img_features, text_features, train_links):
        """
        计算CLIP风格的对比学习损失
        
        Args:
            img_features: 图像特征 [N, D]
            text_features: 文本特征 [N, D]
            train_links: 训练链接 [M, 2]
        """
        if img_features is None or text_features is None:
            return torch.tensor(0.0, device=self.device)
        
        # 归一化特征
        img_features = F.normalize(img_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 提取对齐实体的特征
        img_train = img_features[train_links[:, 0]]  # 左侧KG的图像特征
        text_train = text_features[train_links[:, 1]]  # 右侧KG的文本特征
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(img_train, text_train.T) / self.temperature
        
        # 创建标签（对角线为正样本）
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=self.device)
        
        # 计算双向对比损失
        loss_i2t = self.ce_loss(sim_matrix, labels)
        loss_t2i = self.ce_loss(sim_matrix.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


class KnowledgeAwareCLIPLoss(nn.Module):
    """知识感知的CLIP损失函数"""
    
    def __init__(self, device, temperature=0.07, knowledge_weight=0.1):
        super(KnowledgeAwareCLIPLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.knowledge_weight = knowledge_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, img_emb, text_emb, entity_names, train_ill, graph_emb=None):
        """
        知识感知的跨模态对齐损失
        
        Args:
            img_emb: 图像嵌入 [N, D]
            text_emb: 文本嵌入 [N, D]
            entity_names: 实体名称嵌入 [N, D]
            train_ill: 训练对齐数据 [M, 2]
            graph_emb: 图结构嵌入 [N, D] (可选)
        """
        if img_emb is None or text_emb is None:
            return torch.tensor(0.0, device=self.device)
        
        total_loss = 0.0
        loss_count = 0
        
        # 1. 图像-文本对齐损失
        for i, (e1, e2) in enumerate(train_ill):
            # 图像-文本对齐
            img_text_sim = F.cosine_similarity(img_emb[e1], text_emb[e2], dim=0)
            
            # 跨KG实体对齐
            cross_kg_sim = F.cosine_similarity(img_emb[e1], img_emb[e2], dim=0)
            cross_kg_sim += F.cosine_similarity(text_emb[e1], text_emb[e2], dim=0)
            
            # 组合损失
            alignment_loss = -torch.log(torch.sigmoid(img_text_sim + cross_kg_sim))
            total_loss += alignment_loss
            loss_count += 1
        
        # 2. 实体名称一致性损失（如果提供）
        if entity_names is not None:
            for i, (e1, e2) in enumerate(train_ill):
                name_sim = F.cosine_similarity(entity_names[e1], entity_names[e2], dim=0)
                name_loss = -torch.log(torch.sigmoid(name_sim))
                total_loss += self.knowledge_weight * name_loss
                loss_count += 1
        
        # 3. 图结构一致性损失（如果提供）
        if graph_emb is not None:
            for i, (e1, e2) in enumerate(train_ill):
                graph_sim = F.cosine_similarity(graph_emb[e1], graph_emb[e2], dim=0)
                graph_loss = -torch.log(torch.sigmoid(graph_sim))
                total_loss += self.knowledge_weight * graph_loss
                loss_count += 1
        
        return total_loss / max(loss_count, 1)


class VariationalCLIPLoss(nn.Module):
    """变分CLIP损失函数"""
    
    def __init__(self, device, temperature=0.07, kl_weight=0.01):
        super(VariationalCLIPLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.clip_loss = CLIPAlignmentLoss(device, temperature)
    
    def kl_divergence(self, mu1, logvar1, mu2, logvar2):
        """计算两个高斯分布之间的KL散度"""
        return 0.5 * (
            logvar2 - logvar1 - 1 + 
            ((logvar1.exp() + (mu1 - mu2).pow(2)) / logvar2.exp())
        ).sum()
    
    def forward(self, clip_img_mu, clip_img_logvar, clip_text_mu, clip_text_logvar, train_links):
        """
        变分CLIP损失
        
        Args:
            clip_img_mu: CLIP图像特征均值
            clip_img_logvar: CLIP图像特征对数方差
            clip_text_mu: CLIP文本特征均值  
            clip_text_logvar: CLIP文本特征对数方差
            train_links: 训练链接
        """
        # 重参数化采样
        clip_img_std = torch.exp(0.5 * clip_img_logvar)
        clip_text_std = torch.exp(0.5 * clip_text_logvar)
        
        img_eps = torch.randn_like(clip_img_std)
        text_eps = torch.randn_like(clip_text_std)
        
        clip_img_z = clip_img_mu + img_eps * clip_img_std
        clip_text_z = clip_text_mu + text_eps * clip_text_std
        
        # CLIP对齐损失
        alignment_loss = self.clip_loss(clip_img_z, clip_text_z, train_links)
        
        # 跨模态KL散度
        kl_loss = self.kl_divergence(
            clip_img_mu, clip_img_logvar,
            clip_text_mu, clip_text_logvar
        )
        
        return alignment_loss + self.kl_weight * kl_loss
