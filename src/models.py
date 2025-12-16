#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的多模态实体对齐模型
主要改进：
1. 模态可靠性评估 - 自动检测缺失/低质量模态
2. 自适应融合权重 - 实体级别的动态权重
3. 跨模态对齐机制 - 强制模态间一致性
4. 图像特征增强 - 更好的缺失处理策略
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

# CLIP相关导入
try:
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available, CLIP features disabled")
    CLIPModel = CLIPProcessor = CLIPTokenizer = None
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
    """修复的GAT实现 - 正确处理多头注意力输出"""
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
            n_head = self.n_heads[i] if i < len(self.n_heads) else 1
            
            if x.size(0) > 20000:
                x = self.simple_forward(x, adj, layer, n_head)
            else:
                x = layer(x, adj)  # 返回 [n_head, N, f_out]
                
                # 🔧 修复：正确处理多头输出维度
                if x.dim() == 3 and x.size(0) == n_head:
                    # 方案1: 多头平均
                    x = x.mean(dim=0)  # [N, f_out]
                    # 方案2: 或者拼接后投影 (需要额外参数)
                    # x = x.transpose(0, 1).contiguous().view(N, -1)
                
            if hasattr(self, 'norm_layers'):
                x = self.norm_layers[i](x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def simple_forward(self, x, adj, layer, n_head):
        """简化的前向传播，避免内存爆炸"""
        w = layer.w[0] if layer.diag else layer.w[0]
        
        if layer.diag:
            h = torch.mul(x, w)
        else:
            h = torch.mm(x, w)
        
        if adj.is_sparse:
            output = torch.sparse.mm(adj, h)
        else:
            output = torch.mm(adj, h)
        
        return output


class CLIPEntityEncoder(nn.Module):
    """
    修复的CLIP实体编码器
    
    🔧 修复：
    1. 添加 encode_images_from_raw() 真正使用 CLIP 的 ViT 编码器
    2. 改进 encode_images() 使用多层对齐网络（而非简单线性层）
    3. 添加图像-文本相似度计算
    """
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", freeze_clip=True, device=None):
        super(CLIPEntityEncoder, self).__init__()
        
        if not CLIP_AVAILABLE:
            raise ImportError("transformers library required for CLIP")
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        self.clip_model = self.clip_model.to(self.device)
        
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        self.clip_dim = self.clip_model.config.projection_dim
        self.image_projection = None
        self.feature_aligner = None  # 🆕 多层特征对齐网络
    
    def encode_images_from_raw(self, images):
        """
        🆕 从原始图像编码（真正使用ViT！）
        
        Args:
            images: PIL Images 列表或已处理的 pixel_values tensor
        
        Returns:
            image_features: [N, clip_dim] 归一化的图像特征
        """
        try:
            if isinstance(images, list):
                inputs = self.clip_processor(images=images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
            else:
                pixel_values = images.to(self.device)
            
            # 🔥 真正使用 CLIP 的 ViT 编码器！
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
            
            return F.normalize(image_features, dim=-1)
        
        except Exception as e:
            print(f"Warning: Failed to encode raw images with ViT: {e}")
            return None
    
    def encode_images(self, image_features, use_alignment=True):
        """
        编码预提取的图像特征
        
        🔧 改进：使用多层感知机对齐到CLIP空间（而非简单线性层）
        """
        input_dim = image_features.size(-1)
        
        if use_alignment:
            if self.feature_aligner is None or self.feature_aligner[0].in_features != input_dim:
                # 多层对齐网络，比单个线性层效果更好
                self.feature_aligner = nn.Sequential(
                    nn.Linear(input_dim, self.clip_dim * 2),
                    nn.LayerNorm(self.clip_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.clip_dim * 2, self.clip_dim),
                    nn.LayerNorm(self.clip_dim)
                ).to(self.device)
            
            aligned = self.feature_aligner(image_features)
        else:
            if self.image_projection is None or self.image_projection.in_features != input_dim:
                self.image_projection = nn.Linear(input_dim, self.clip_dim).to(self.device)
            aligned = self.image_projection(image_features)
        
        return F.normalize(aligned, dim=-1)
    
    def encode_texts(self, texts):
        """编码文本（使用CLIP的文本编码器）"""
        if isinstance(texts, str):
            texts = [texts]
        
        if len(texts) > 1000:
            batch_size = 1000
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self._encode_text_batch(batch_texts)
                all_embeddings.append(batch_embeddings)
            return torch.cat(all_embeddings, dim=0)
        else:
            return self._encode_text_batch(texts)
    
    def _encode_text_batch(self, texts):
        """编码一批文本"""
        inputs = self.clip_tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=77
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_outputs = self.clip_model.get_text_features(**inputs)
        
        return F.normalize(text_outputs, dim=-1)
    
    def compute_similarity(self, image_features, text_features):
        """🆕 计算图像-文本相似度"""
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return torch.matmul(image_features, text_features.T)
    
    def forward(self, image_features=None, texts=None, raw_images=None):
        """
        前向传播
        
        Args:
            image_features: 预提取的图像特征（可选）
            texts: 实体文本（可选）
            raw_images: 原始图像（可选，如果提供则使用ViT编码）
        """
        results = {}
        
        # 优先使用原始图像（真正的ViT编码）
        if raw_images is not None:
            vit_features = self.encode_images_from_raw(raw_images)
            if vit_features is not None:
                results['image_embeds'] = vit_features
                results['used_vit'] = True
        
        # 如果没有原始图像，使用预提取特征
        if 'image_embeds' not in results and image_features is not None:
            results['image_embeds'] = self.encode_images(image_features)
            results['used_vit'] = False
        
        if texts is not None:
            results['text_embeds'] = self.encode_texts(texts)
        
        return results


class ModalityReliabilityEstimator(nn.Module):
    """
    模态可靠性评估器
    为每个实体的每个模态估计一个可靠性分数
    """
    
    def __init__(self, modal_dims, hidden_dim=64):
        super(ModalityReliabilityEstimator, self).__init__()
        self.modal_dims = modal_dims
        
        # 为每个模态创建可靠性评估网络
        self.reliability_nets = nn.ModuleDict()
        for modal_name, modal_dim in modal_dims.items():
            self.reliability_nets[modal_name] = nn.Sequential(
                nn.Linear(modal_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # 使用普通属性存储统计量（不能用register_buffer存储字典）
        self.modal_means = {}
        self.modal_stds = {}
    
    def compute_feature_quality(self, features, modal_name):
        """
        计算特征质量分数
        基于：
        1. 特征方差（随机特征通常方差较小）
        2. 特征范数（正常特征有合理的范数）
        3. 网络预测的可靠性
        """
        if features is None:
            return torch.zeros(1, device=next(self.parameters()).device)
        
        # 1. 计算特征统计量
        feature_var = features.var(dim=-1, keepdim=True)  # [N, 1]
        feature_norm = features.norm(dim=-1, keepdim=True)  # [N, 1]
        
        # 2. 归一化特征
        features_normalized = F.normalize(features, dim=-1)
        
        # 3. 网络预测
        if modal_name in self.reliability_nets:
            reliability = self.reliability_nets[modal_name](features)  # [N, 1]
        else:
            reliability = torch.ones(features.size(0), 1, device=features.device)
        
        # 4. 组合多个信号
        # 低方差的特征可能是随机填充的
        var_score = torch.sigmoid(feature_var * 10 - 0.5)  # 方差太小则分数低
        norm_score = torch.sigmoid(feature_norm - 0.5)  # 范数太小则分数低
        
        quality_score = (reliability + var_score + norm_score) / 3.0
        
        return quality_score.squeeze(-1)
    
    def forward(self, modal_features):
        """
        计算所有模态的可靠性分数
        
        Args:
            modal_features: dict, {modal_name: features [N, D]}
        
        Returns:
            reliability_scores: dict, {modal_name: scores [N]}
        """
        reliability_scores = {}
        
        for modal_name, features in modal_features.items():
            if features is not None:
                reliability_scores[modal_name] = self.compute_feature_quality(
                    features, modal_name
                )
            else:
                reliability_scores[modal_name] = None
        
        return reliability_scores


class AdaptiveMultiModalFusion(nn.Module):
    """
    自适应多模态融合层
    改进点：
    1. 实体级别的动态权重（不是全局权重）
    2. 基于可靠性的加权
    3. 跨模态注意力交互
    """
    
    def __init__(self, modal_dims, output_dim, use_cross_modal_attention=True):
        super(AdaptiveMultiModalFusion, self).__init__()
        self.modal_dims = modal_dims
        self.output_dim = output_dim
        self.use_cross_modal_attention = use_cross_modal_attention
        
        # 标准化维度
        self.standard_dim = 128
        
        # 模态投影层
        self.modal_projections = nn.ModuleDict()
        for modal_name, modal_dim in modal_dims.items():
            self.modal_projections[modal_name] = nn.Sequential(
                nn.Linear(modal_dim, self.standard_dim),
                nn.LayerNorm(self.standard_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # 可靠性评估器
        self.reliability_estimator = ModalityReliabilityEstimator(modal_dims)
        
        # 自适应门控网络（生成实体级别的模态权重）
        num_modals = len(modal_dims)
        self.gate_network = nn.Sequential(
            nn.Linear(num_modals * self.standard_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_modals),
            nn.Softmax(dim=-1)
        )
        
        # 跨模态注意力（可选）
        if use_cross_modal_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.standard_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
        
        # 最终投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.standard_dim * num_modals, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 残差连接的投影
        self.residual_projections = nn.ModuleDict()
        for modal_name in modal_dims.keys():
            self.residual_projections[modal_name] = nn.Linear(self.standard_dim, output_dim)
    
    def forward(self, modal_features, knowledge_context=None):
        """
        自适应融合多模态特征
        
        Args:
            modal_features: dict, {modal_name: features [N, D]}
            knowledge_context: 图结构上下文（可选）
        
        Returns:
            fused_features: [N, output_dim]
        """
        device = next(self.parameters()).device
        
        # 1. 投影所有模态到标准维度
        projected_features = {}
        valid_modals = []
        
        for modal_name in self.modal_dims.keys():
            if modal_name in modal_features and modal_features[modal_name] is not None:
                feat = modal_features[modal_name].to(device)
                projected = self.modal_projections[modal_name](feat)
                projected_features[modal_name] = projected
                valid_modals.append(modal_name)
        
        if not valid_modals:
            # 没有有效模态，返回零向量
            batch_size = 1
            if knowledge_context is not None:
                batch_size = knowledge_context.size(0)
            return torch.zeros(batch_size, self.output_dim, device=device)
        
        # 确保batch size一致
        batch_size = min(feat.size(0) for feat in projected_features.values())
        for modal_name in projected_features:
            projected_features[modal_name] = projected_features[modal_name][:batch_size]
        
        # 2. 计算模态可靠性分数
        reliability_scores = self.reliability_estimator(modal_features)
        
        # 3. 如果使用跨模态注意力，进行特征交互
        if self.use_cross_modal_attention and len(valid_modals) > 1:
            # 将所有模态特征堆叠为序列 [N, num_modals, D]
            modal_stack = torch.stack([projected_features[m] for m in valid_modals], dim=1)
            
            # 跨模态注意力
            attended, _ = self.cross_attention(modal_stack, modal_stack, modal_stack)
            
            # 更新投影特征
            for i, modal_name in enumerate(valid_modals):
                projected_features[modal_name] = attended[:, i, :]
        
        # 4. 计算自适应门控权重
        # 拼接所有投影特征
        concat_features = []
        for modal_name in self.modal_dims.keys():
            if modal_name in projected_features:
                concat_features.append(projected_features[modal_name])
            else:
                # 对于缺失模态，使用零向量
                concat_features.append(torch.zeros(batch_size, self.standard_dim, device=device))
        
        concat_features = torch.cat(concat_features, dim=-1)  # [N, num_modals * D]
        gate_weights = self.gate_network(concat_features)  # [N, num_modals]
        
        # 5. 结合可靠性分数调整权重
        adjusted_weights = []
        for i, modal_name in enumerate(self.modal_dims.keys()):
            if modal_name in reliability_scores and reliability_scores[modal_name] is not None:
                rel_score = reliability_scores[modal_name][:batch_size]
                adjusted_weight = gate_weights[:, i] * rel_score
            else:
                adjusted_weight = gate_weights[:, i] * 0.1  # 缺失模态给予很小权重
            adjusted_weights.append(adjusted_weight)
        
        adjusted_weights = torch.stack(adjusted_weights, dim=-1)  # [N, num_modals]
        adjusted_weights = F.softmax(adjusted_weights, dim=-1)  # 重新归一化
        
        # 6. 加权融合
        weighted_features = []
        for i, modal_name in enumerate(self.modal_dims.keys()):
            if modal_name in projected_features:
                weighted = projected_features[modal_name] * adjusted_weights[:, i:i+1]
            else:
                weighted = torch.zeros(batch_size, self.standard_dim, device=device)
            weighted_features.append(weighted)
        
        # 拼接加权特征
        fused = torch.cat(weighted_features, dim=-1)  # [N, num_modals * D]
        
        # 7. 最终投影
        output = self.output_projection(fused)
        
        # 8. 残差连接（从最可靠的模态）
        if knowledge_context is not None and 'graph' in projected_features:
            residual = self.residual_projections['graph'](projected_features['graph'])
            output = output + 0.1 * residual
        
        return output
    
    def get_modal_weights(self, modal_features):
        """获取当前的模态权重（用于分析）"""
        device = next(self.parameters()).device
        
        projected_features = {}
        for modal_name in self.modal_dims.keys():
            if modal_name in modal_features and modal_features[modal_name] is not None:
                feat = modal_features[modal_name].to(device)
                projected = self.modal_projections[modal_name](feat)
                projected_features[modal_name] = projected
        
        batch_size = min(feat.size(0) for feat in projected_features.values())
        
        concat_features = []
        for modal_name in self.modal_dims.keys():
            if modal_name in projected_features:
                concat_features.append(projected_features[modal_name][:batch_size])
            else:
                concat_features.append(torch.zeros(batch_size, self.standard_dim, device=device))
        
        concat_features = torch.cat(concat_features, dim=-1)
        gate_weights = self.gate_network(concat_features)
        
        return gate_weights


class CrossModalAlignmentModule(nn.Module):
    """
    跨模态对齐模块
    强制不同模态表示同一实体时的一致性
    """
    
    def __init__(self, modal_dims, align_dim=128):
        super(CrossModalAlignmentModule, self).__init__()
        self.modal_dims = modal_dims
        self.align_dim = align_dim
        
        # 为每个模态创建对齐投影
        self.align_projections = nn.ModuleDict()
        for modal_name, modal_dim in modal_dims.items():
            self.align_projections[modal_name] = nn.Sequential(
                nn.Linear(modal_dim, align_dim),
                nn.LayerNorm(align_dim)
            )
    
    def compute_alignment_loss(self, modal_features, train_links):
        """
        计算跨模态对齐损失
        
        对于对齐的实体对(e1, e2)，它们在不同模态中的表示应该相似
        """
        device = next(self.parameters()).device
        
        # 收集有效模态
        valid_modals = {k: v for k, v in modal_features.items() if v is not None}
        
        if len(valid_modals) < 2:
            return torch.tensor(0.0, device=device)
        
        total_loss = 0.0
        pair_count = 0
        
        modal_names = list(valid_modals.keys())
        
        # 计算每对模态之间的对齐损失
        for i in range(len(modal_names)):
            for j in range(i + 1, len(modal_names)):
                modal_i = modal_names[i]
                modal_j = modal_names[j]
                
                feat_i = valid_modals[modal_i]
                feat_j = valid_modals[modal_j]
                
                # 投影到对齐空间
                proj_i = self.align_projections[modal_i](feat_i)
                proj_j = self.align_projections[modal_j](feat_j)
                
                # 归一化
                proj_i = F.normalize(proj_i, dim=-1)
                proj_j = F.normalize(proj_j, dim=-1)
                
                # 确保索引有效
                min_size = min(proj_i.size(0), proj_j.size(0))
                valid_links = train_links[train_links[:, 0] < min_size]
                valid_links = valid_links[valid_links[:, 1] < min_size]
                
                if len(valid_links) == 0:
                    continue
                
                # 对齐实体应该在不同模态中相似
                aligned_i = proj_i[valid_links[:, 0]]
                aligned_j = proj_j[valid_links[:, 1]]
                
                # 计算余弦相似度损失（鼓励相似）
                similarity = F.cosine_similarity(aligned_i, aligned_j, dim=-1)
                alignment_loss = 1.0 - similarity.mean()
                
                total_loss += alignment_loss
                pair_count += 1
        
        if pair_count > 0:
            return total_loss / pair_count
        else:
            return torch.tensor(0.0, device=device)


class IBMultiModal(nn.Module):
    """
    改进的多模态实体对齐模型
    主要改进：
    1. 自适应多模态融合
    2. 模态可靠性评估
    3. 跨模态对齐
    """
    
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

        # 特征处理层 - 改进的投影
        self.rel_fc = nn.Sequential(
            nn.Linear(1000, attr_dim * 2),
            nn.LayerNorm(attr_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attr_dim * 2, attr_dim)
        )
        
        self.att_fc = nn.Sequential(
            nn.Linear(1000, attr_dim * 2),
            nn.LayerNorm(attr_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attr_dim * 2, attr_dim)
        )
        
        self.img_fc = nn.Sequential(
            nn.Linear(img_feature_dim, img_dim * 2),
            nn.LayerNorm(img_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(img_dim * 2, img_dim)
        )

        if char_feature_dim is None:
            char_feature_dim = 100
        self.name_fc = nn.Linear(300, char_dim)
        self.char_fc = nn.Linear(char_feature_dim, char_dim)

        # CLIP相关组件
        self.use_clip = getattr(args, 'use_clip', False)
        self.clip_encoder = None
        self.clip_dim = 0
        
        if self.use_clip and CLIP_AVAILABLE:
            try:
                self.clip_encoder = CLIPEntityEncoder(
                    clip_model_name=getattr(args, 'clip_model_name', 'openai/clip-vit-base-patch32'),
                    freeze_clip=getattr(args, 'clip_freeze', True),
                    device=getattr(args, 'device', None)
                )
                self.clip_dim = self.clip_encoder.clip_dim
                self.clip_projection = nn.Linear(self.clip_dim, img_dim)
                print(f"CLIP encoder initialized with dimension {self.clip_dim}")
            except Exception as e:
                print(f"Warning: Failed to initialize CLIP encoder: {e}")
                self.use_clip = False
                self.clip_encoder = None
                self.clip_dim = 0

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
        self._init_structure_encoder(dropout, diag)

        # 多模态融合 - 使用改进的自适应融合
        modal_dims = {}
        if args.w_gcn: modal_dims['graph'] = self.n_units[-1]
        if args.w_img: modal_dims['image'] = img_dim
        if args.w_rel: modal_dims['relation'] = attr_dim  
        if args.w_attr: modal_dims['attribute'] = attr_dim
        if args.w_name: modal_dims['name'] = char_dim
        if args.w_char: modal_dims['char'] = char_dim
        if self.use_clip: modal_dims['clip'] = img_dim
        
        # 使用改进的自适应融合层
        joint_dim = getattr(args, 'joint_dim', 600)
        self.fusion = AdaptiveMultiModalFusion(
            modal_dims=modal_dims,
            output_dim=joint_dim,
            use_cross_modal_attention=getattr(args, 'use_cross_modal_attention', True)
        )
        
        # 跨模态对齐模块
        self.cross_modal_alignment = CrossModalAlignmentModule(
            modal_dims=modal_dims,
            align_dim=128
        )

        # 投影头
        if self.use_project_head:
            self.img_pro = ProjectionHead(img_dim, img_dim, img_dim, dropout)
            self.att_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.rel_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.gph_pro = ProjectionHead(self.n_units[-1], self.n_units[-1], self.n_units[-1], dropout)

        # 联合变分编码
        if self.use_joint_vib:
            self.joint_fc = nn.Linear(joint_dim, joint_dim)
            self.joint_fc_mu = nn.Linear(joint_dim, joint_dim)
            self.joint_fc_std = nn.Linear(joint_dim, joint_dim)

        # KLD损失记录
        self.kld_loss = 0.0
        self.img_kld_loss = 0.0
        self.rel_kld_loss = 0.0
        self.attr_kld_loss = 0.0
        self.joint_kld_loss = 0.0
        
        # CLIP特征记录
        self.clip_features = {}
        
        # 模态特征记录（用于跨模态对齐损失）
        self.modal_features = {}

    def _init_structure_encoder(self, dropout, diag):
        """初始化图结构编码器"""
        structure_encoder = getattr(self.args, 'structure_encoder', 'gcn')
        
        if structure_encoder == 'gcn':
            if self.use_graph_vib:
                # 🆕 Graph VIB: 创建mu和logvar两个网络
                self.cross_graph_model_mu = SimpleGCN(
                    self.input_dim, 
                    self.n_units[1] if len(self.n_units) > 1 else self.input_dim,
                    self.n_units[-1], 
                    dropout
                )
                self.cross_graph_model_logvar = SimpleGCN(
                    self.input_dim, 
                    self.n_units[1] if len(self.n_units) > 1 else self.input_dim,
                    self.n_units[-1], 
                    dropout
                )
            else:
                self.cross_graph_model = SimpleGCN(
                    self.input_dim, 
                    self.n_units[1] if len(self.n_units) > 1 else self.input_dim,
                    self.n_units[-1], 
                    dropout
                )
        elif structure_encoder == 'gat':
            if self.use_graph_vib:
                # 🆕 GAT + VIB
                self.cross_graph_model_mu = SimpleGAT(
                    self.n_units,
                    self.n_heads,
                    dropout,
                    getattr(self.args, 'attn_dropout', 0.0),
                    getattr(self.args, 'instance_normalization', False),
                    diag
                )
                self.cross_graph_model_logvar = SimpleGAT(
                    self.n_units,
                    self.n_heads,
                    dropout,
                    getattr(self.args, 'attn_dropout', 0.0),
                    getattr(self.args, 'instance_normalization', False),
                    diag
                )
            else:
                self.cross_graph_model = SimpleGAT(
                    self.n_units,
                    self.n_heads,
                    dropout,
                    getattr(self.args, 'attn_dropout', 0.0),
                    getattr(self.args, 'instance_normalization', False),
                    diag
                )
        else:
            self.cross_graph_model = SimpleGCN(
                self.input_dim,
                self.n_units[1] if len(self.n_units) > 1 else self.input_dim,
                self.n_units[-1],
                dropout
            )

    def _kld_gauss(self, mu_q, std_q, mu_p, std_p):
        """计算两个高斯分布之间的KL散度"""
        kld = (
            torch.log(std_p / (std_q + 1e-8) + 1e-8)
            + (std_q ** 2 + (mu_q - mu_p) ** 2) / (2 * std_p ** 2 + 1e-8)
            - 0.5
        )
        return kld.mean()

    def forward(
        self,
        input_idx,
        adj,
        img_features=None,
        rel_features=None,
        att_features=None,
        name_features=None,
        char_features=None,
        entity_texts=None,
    ):
        """
        前向传播
        """
        modal_features = {}
        
        # === 图结构特征处理 ===
        gph_emb = None
        if self.args.w_gcn:
            try:
                entity_input = self.entity_emb(input_idx)
                
                if self.use_graph_vib and hasattr(self, 'cross_graph_model_mu'):
                    # 🆕 Graph VIB: 使用正确的reparameterization
                    gph_mu = self.cross_graph_model_mu(entity_input, adj)
                    gph_logvar = self.cross_graph_model_logvar(entity_input, adj)
                    # 约束logvar范围，防止数值不稳定
                    gph_logvar = torch.clamp(gph_logvar, -10, 2)
                    gph_std = torch.exp(0.5 * gph_logvar)
                    eps = torch.randn_like(gph_std)
                    gph_emb = gph_mu + eps * gph_std
                    # 计算KLD损失
                    self.kld_loss = self._kld_gauss(
                        gph_mu, gph_std, 
                        torch.zeros_like(gph_mu), torch.ones_like(gph_std)
                    )
                else:
                    gph_emb = self.cross_graph_model(entity_input, adj)
                    
                modal_features['graph'] = gph_emb
            except Exception as e:
                print(f"Warning: Graph encoding failed: {e}")
                entity_input = self.entity_emb(input_idx)
                gph_emb = entity_input
                modal_features['graph'] = gph_emb

        # === 图像特征处理 ===
        img_emb = None
        if self.args.w_img and img_features is not None:
            try:
                img_emb = self.img_fc(img_features)
                
                if self.use_img_vib and hasattr(self, 'img_fc_mu'):
                    img_emb_h = F.relu(img_emb)
                    mu = self.img_fc_mu(img_emb_h)
                    logvar = self.img_fc_std(img_emb_h)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    img_emb = mu + eps * std
                    self.img_kld_loss = self._kld_gauss(
                        mu, std, torch.zeros_like(mu), torch.ones_like(std)
                    )
                
                modal_features['image'] = img_emb
            except Exception as e:
                print(f"Warning: Image encoding failed: {e}")

        # === 关系特征处理 ===
        rel_emb = None
        if self.args.w_rel and rel_features is not None:
            try:
                rel_emb = self.rel_fc(rel_features)
                
                if self.use_rel_vib and hasattr(self, 'rel_fc_mu'):
                    rel_emb_h = F.relu(rel_emb)
                    mu = self.rel_fc_mu(rel_emb_h)
                    logvar = self.rel_fc_std(rel_emb_h)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    rel_emb = mu + eps * std
                    self.rel_kld_loss = self._kld_gauss(
                        mu, std, torch.zeros_like(mu), torch.ones_like(std)
                    )
                
                modal_features['relation'] = rel_emb
            except Exception as e:
                print(f"Warning: Relation encoding failed: {e}")

        # === 属性特征处理 ===
        att_emb = None
        if self.args.w_attr and att_features is not None:
            try:
                att_emb = self.att_fc(att_features)
                
                if self.use_attr_vib and hasattr(self, 'att_fc_mu'):
                    att_emb_h = F.relu(att_emb)
                    mu = self.att_fc_mu(att_emb_h)
                    logvar = self.att_fc_std(att_emb_h)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    att_emb = mu + eps * std
                    self.attr_kld_loss = self._kld_gauss(
                        mu, std, torch.zeros_like(mu), torch.ones_like(std)
                    )
                
                modal_features['attribute'] = att_emb
            except Exception as e:
                print(f"Warning: Attribute encoding failed: {e}")

        # === 名称和字符特征处理 ===
        name_emb = None
        if self.args.w_name and name_features is not None:
            try:
                name_emb = self.name_fc(name_features)
                modal_features['name'] = name_emb
            except Exception as e:
                print(f"Warning: Name encoding failed: {e}")
            
        char_emb = None
        if self.args.w_char and char_features is not None:
            try:
                char_emb = self.char_fc(char_features)
                modal_features['char'] = char_emb
            except Exception as e:
                print(f"Warning: Char encoding failed: {e}")

        # === CLIP特征处理 ===
        if self.use_clip and self.clip_encoder is not None:
            try:
                clip_results = {}
                
                if img_features is not None:
                    clip_results['image_embeds'] = self.clip_encoder.encode_images(img_features)
                    self.clip_features['image_embeds'] = clip_results['image_embeds']
                
                if entity_texts is not None:
                    clip_results['text_embeds'] = self.clip_encoder.encode_texts(entity_texts)
                    self.clip_features['text_embeds'] = clip_results['text_embeds']
                
                if 'image_embeds' in clip_results and 'text_embeds' in clip_results:
                    clip_fused = (clip_results['image_embeds'] + clip_results['text_embeds']) / 2
                    clip_projected = self.clip_projection(clip_fused)
                    modal_features['clip'] = clip_projected
                elif 'image_embeds' in clip_results:
                    clip_projected = self.clip_projection(clip_results['image_embeds'])
                    modal_features['clip'] = clip_projected
                elif 'text_embeds' in clip_results:
                    clip_projected = self.clip_projection(clip_results['text_embeds'])
                    modal_features['clip'] = clip_projected
                    
            except Exception as e:
                print(f"Warning: CLIP encoding failed: {e}")

        # === 投影头处理 ===
        if self.use_project_head:
            if 'image' in modal_features and modal_features['image'] is not None:
                modal_features['image'] = self.img_pro(modal_features['image'])
            if 'attribute' in modal_features and modal_features['attribute'] is not None:
                modal_features['attribute'] = self.att_pro(modal_features['attribute'])
            if 'relation' in modal_features and modal_features['relation'] is not None:
                modal_features['relation'] = self.rel_pro(modal_features['relation'])
            if 'graph' in modal_features and modal_features['graph'] is not None:
                modal_features['graph'] = self.gph_pro(modal_features['graph'])

        # 保存模态特征（用于跨模态对齐损失）
        self.modal_features = modal_features

        # === 多模态融合 ===
        try:
            knowledge_context = modal_features.get('graph', None)
            joint_emb = self.fusion(modal_features, knowledge_context)
        except Exception as e:
            print(f"Warning: Fusion failed: {e}")
            joint_emb = self._fallback_fusion(modal_features, input_idx)

        # === 联合变分编码 ===
        if self.use_joint_vib and hasattr(self, 'joint_fc_mu'):
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

    def _fallback_fusion(self, modal_features, input_idx):
        """备用融合方法"""
        valid_embeddings = [emb for emb in modal_features.values() if emb is not None]
        
        if valid_embeddings:
            standard_dim = 128
            projected_embeddings = []
            
            for i, emb in enumerate(valid_embeddings):
                proj_attr_name = f'fallback_projection_{i}'
                if not hasattr(self, proj_attr_name):
                    setattr(self, proj_attr_name, 
                           nn.Linear(emb.size(-1), standard_dim).to(emb.device))
                proj_layer = getattr(self, proj_attr_name)
                projected_embeddings.append(proj_layer(emb))
            
            joint_emb = torch.cat(projected_embeddings, dim=-1)
            
            if not hasattr(self, 'final_fallback_projection'):
                input_dim = joint_emb.size(-1)
                output_dim = getattr(self.args, 'joint_dim', 600)
                self.final_fallback_projection = nn.Linear(input_dim, output_dim).to(joint_emb.device)
            joint_emb = self.final_fallback_projection(joint_emb)
        else:
            entity_input = self.entity_emb(input_idx)
            output_dim = getattr(self.args, 'joint_dim', 600)
            if not hasattr(self, 'emergency_projection'):
                self.emergency_projection = nn.Linear(entity_input.size(-1), output_dim).to(entity_input.device)
            joint_emb = self.emergency_projection(entity_input)
        
        return joint_emb
    
    def get_cross_modal_alignment_loss(self, train_links):
        """获取跨模态对齐损失"""
        if hasattr(self, 'modal_features') and self.modal_features:
            return self.cross_modal_alignment.compute_alignment_loss(
                self.modal_features, train_links
            )
        return torch.tensor(0.0)


# 为了保持兼容性，设置别名
KAMultiModal = IBMultiModal
ClipEnhancedKACEA = IBMultiModal
KACEA = IBMultiModal