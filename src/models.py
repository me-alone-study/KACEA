#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    """修正的CLIP实体编码器"""
    
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", freeze_clip=True, device=None):
        super(CLIPEntityEncoder, self).__init__()
        
        if not CLIP_AVAILABLE:
            raise ImportError("transformers library required for CLIP")
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载预训练的CLIP模型
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # 移动到指定设备
        self.clip_model = self.clip_model.to(self.device)
        
        # 是否冻结CLIP参数
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # CLIP输出维度
        self.clip_dim = self.clip_model.config.projection_dim  # 通常是512
        
        # 图像特征投影层 - 延迟初始化
        self.image_projection = None
        
    def encode_images(self, image_features):
        """编码图像特征 - 修复维度匹配问题"""
        # 创建自适应投影层
        input_dim = image_features.size(-1)
        if self.image_projection is None or self.image_projection.in_features != input_dim:
            self.image_projection = nn.Linear(input_dim, self.clip_dim).to(self.device)
        
        projected = self.image_projection(image_features)
        return F.normalize(projected, dim=-1)
    
    def encode_texts(self, texts):
        """编码文本"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 处理批量文本
        if len(texts) > 1000:  # 如果文本太多，分批处理
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
        # 使用CLIP tokenizer处理文本
        inputs = self.clip_tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=77
        )
        
        # 移动到正确的设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 获取文本嵌入
        with torch.no_grad():
            text_outputs = self.clip_model.get_text_features(**inputs)
        
        return F.normalize(text_outputs, dim=-1)
    
    def forward(self, image_features=None, texts=None):
        """前向传播"""
        results = {}
        
        if image_features is not None:
            results['image_embeds'] = self.encode_images(image_features)
        
        if texts is not None:
            results['text_embeds'] = self.encode_texts(texts)
        
        return results


class KnowledgeAwareFusion(nn.Module):
    """修正的知识感知特征融合层"""
    
    def __init__(self, modal_dims, output_dim, fusion_strategy="weighted_concat"):
        super(KnowledgeAwareFusion, self).__init__()
        self.fusion_strategy = fusion_strategy
        self.modal_dims = modal_dims
        self.output_dim = output_dim
        
        # 计算标准化维度 - 确保所有模态特征都投影到相同维度
        self.standard_dim = 128  # 标准化维度
        
        # 为每个模态创建投影层
        self.modal_projections = nn.ModuleDict()
        for modal_name, modal_dim in modal_dims.items():
            self.modal_projections[modal_name] = nn.Linear(modal_dim, self.standard_dim)
        
        if fusion_strategy == "weighted_concat":
            # 为每个模态学习权重
            self.modal_weights = nn.Parameter(torch.ones(len(modal_dims)))
            total_dim = len(modal_dims) * self.standard_dim
            self.projection = nn.Linear(total_dim, output_dim)
            
        elif fusion_strategy == "attention":
            # 注意力融合
            total_dim = len(modal_dims) * self.standard_dim
            self.attention = nn.Linear(total_dim, len(modal_dims))
            self.projection = nn.Linear(total_dim, output_dim)
            
        elif fusion_strategy == "simple_concat":
            # 简单拼接
            total_dim = len(modal_dims) * self.standard_dim
            self.projection = nn.Linear(total_dim, output_dim)
    
    def forward(self, modal_features, knowledge_context=None):
        """
        Args:
            modal_features: dict, 包含不同模态的特征
            knowledge_context: 知识上下文信息（如图结构嵌入）
        """
        valid_features = []
        feature_names = []
        
        # 投影所有特征到标准维度
        for name in self.modal_dims.keys():
            if name in modal_features and modal_features[name] is not None:
                # 投影到标准维度
                try:
                    projected_feat = self.modal_projections[name](modal_features[name])
                    valid_features.append(projected_feat)
                    feature_names.append(name)
                except Exception as e:
                    print(f"Warning: Failed to project {name} features: {e}")
                    continue
        
        if not valid_features:
            # 如果没有有效特征，返回零特征
            batch_size = knowledge_context.size(0) if knowledge_context is not None else 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.output_dim, device=device)
        
        # 确保所有特征有相同的batch size
        min_batch_size = min(f.size(0) for f in valid_features)
        valid_features = [f[:min_batch_size] for f in valid_features]
        
        if self.fusion_strategy == "weighted_concat":
            # 加权拼接
            weights = torch.softmax(self.modal_weights[:len(valid_features)], dim=0)
            weighted_features = [w * f for w, f in zip(weights, valid_features)]
            fused = torch.cat(weighted_features, dim=-1)
            
        elif self.fusion_strategy == "attention":
            # 注意力融合
            concat_features = torch.cat(valid_features, dim=-1)
            attention_weights = torch.softmax(self.attention(concat_features), dim=-1)
            
            # 应用注意力权重
            weighted_features = []
            for i, feature in enumerate(valid_features):
                weight = attention_weights[:, i:i+1]
                weighted_features.append(weight * feature)
            
            fused = torch.cat(weighted_features, dim=-1)
            
        else:  # simple_concat
            fused = torch.cat(valid_features, dim=-1)
        
        return self.projection(fused)


class IBMultiModal(nn.Module):
    """修正的多模态实体对齐模型"""
    
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
                
                # CLIP特征投影层
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

        # 图结构编码器 - 修正VIB初始化
        no_diag = getattr(args, 'no_diag', False)
        diag = not no_diag
        
        # 初始化图结构编码器
        self._init_structure_encoder(dropout, diag)

        # 多模态融合 - 修正维度配置
        modal_dims = {}
        if args.w_gcn: modal_dims['graph'] = self.n_units[-1]
        if args.w_img: modal_dims['image'] = img_dim
        if args.w_rel: modal_dims['relation'] = attr_dim  
        if args.w_attr: modal_dims['attribute'] = attr_dim
        if args.w_name: modal_dims['name'] = char_dim
        if args.w_char: modal_dims['char'] = char_dim
        if self.use_clip: modal_dims['clip'] = img_dim  # CLIP特征投影后的维度
        
        # 知识感知融合层
        joint_dim = getattr(args, 'joint_dim', 600)
        self.fusion = KnowledgeAwareFusion(
            modal_dims=modal_dims,
            output_dim=joint_dim,
            fusion_strategy=getattr(args, 'fusion_strategy', 'weighted_concat')
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

        # 初始化损失
        self.kld_loss = 0
        self.img_kld_loss = 0
        self.rel_kld_loss = 0
        self.attr_kld_loss = 0
        self.joint_kld_loss = 0
        self.clip_features = {}  # 存储CLIP特征用于损失计算

    def _init_structure_encoder(self, dropout, diag):
        """初始化图结构编码器"""
        if self.args.structure_encoder == "gcn":
            if self.use_graph_vib:
                self.cross_graph_model_mu = SimpleGCN(
                    self.n_units[0], 
                    self.n_units[1] if len(self.n_units) > 1 else self.n_units[0], 
                    self.n_units[-1], 
                    dropout
                )
                self.cross_graph_model_std = SimpleGCN(
                    self.n_units[0], 
                    self.n_units[1] if len(self.n_units) > 1 else self.n_units[0], 
                    self.n_units[-1], 
                    dropout
                )
            else:
                self.cross_graph_model = SimpleGCN(
                    self.n_units[0], 
                    self.n_units[1] if len(self.n_units) > 1 else self.n_units[0], 
                    self.n_units[-1], 
                    dropout
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
        else:
            # 默认使用GAT
            print(f"Warning: Unknown structure encoder '{self.args.structure_encoder}', using GAT")
            self.args.structure_encoder = "gat"
            self._init_structure_encoder(dropout, diag)

    def _kld_gauss(self, mu_1, logsigma_1, mu_2, logsigma_2):
        """KL散度计算"""
        try:
            from torch.distributions.kl import kl_divergence
            from torch.distributions import Normal
            
            sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp(logsigma_1, -10, 10)))
            sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp(logsigma_2, -10, 10)))
            
            # 处理NaN和inf
            mu_1 = torch.clamp(mu_1, -10, 10)
            mu_2 = torch.clamp(mu_2, -10, 10)
            sigma_1 = torch.clamp(sigma_1, 0.01, 10)
            sigma_2 = torch.clamp(sigma_2, 0.01, 10)
            
            q_target = Normal(mu_1, sigma_1)
            q_context = Normal(mu_2, sigma_2)
            
            kl = kl_divergence(q_target, q_context).mean()
            return torch.clamp(kl, 0, 100)
        except:
            # 简化版本的KL散度
            kl = 0.5 * (logsigma_2 - logsigma_1 - 1.0 + 
                       torch.exp(logsigma_1 - logsigma_2) + 
                       (mu_1 - mu_2) ** 2 * torch.exp(-logsigma_2))
            return torch.clamp(kl.mean(), 0, 100)

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
        modal_features = {}
        self.clip_features = {}  # 重置CLIP特征存储
        
        # === 图结构特征处理 ===
        gph_emb = None
        if self.args.w_gcn:
            try:
                entity_input = self.entity_emb(input_idx)
                if self.use_graph_vib:
                    # 修正：检查属性是否存在
                    if hasattr(self, 'cross_graph_model_mu') and hasattr(self, 'cross_graph_model_std'):
                        gph_emb_mu = self.cross_graph_model_mu(entity_input, adj)
                        gph_emb_std = self.cross_graph_model_std(entity_input, adj)
                        eps = torch.randn_like(gph_emb_std)
                        gph_emb = gph_emb_mu + eps * gph_emb_std
                        self.kld_loss = self._kld_gauss(
                            gph_emb_mu, gph_emb_std, 
                            torch.zeros_like(gph_emb_mu), torch.ones_like(gph_emb_std)
                        )
                    else:
                        print("Warning: VIB graph models not initialized, using standard model")
                        if hasattr(self, 'cross_graph_model'):
                            gph_emb = self.cross_graph_model(entity_input, adj)
                        else:
                            gph_emb = entity_input
                else:
                    if hasattr(self, 'cross_graph_model'):
                        gph_emb = self.cross_graph_model(entity_input, adj)
                    else:
                        gph_emb = entity_input
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
                if self.use_rel_vib and hasattr(self, 'rel_fc_mu'):
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
                modal_features['relation'] = rel_emb
            except Exception as e:
                print(f"Warning: Relation encoding failed: {e}")

        # === 属性特征处理 ===
        att_emb = None
        if self.args.w_attr and att_features is not None:
            try:
                if self.use_attr_vib and hasattr(self, 'att_fc_mu'):
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
                modal_features['attribute'] = att_emb
            except Exception as e:
                print(f"Warning: Attribute encoding failed: {e}")

        # === 其他特征处理 ===
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
                
                # 编码图像特征
                if img_features is not None:
                    clip_results['image_embeds'] = self.clip_encoder.encode_images(img_features)
                    self.clip_features['image_embeds'] = clip_results['image_embeds']
                
                # 编码文本特征
                if entity_texts is not None:
                    clip_results['text_embeds'] = self.clip_encoder.encode_texts(entity_texts)
                    self.clip_features['text_embeds'] = clip_results['text_embeds']
                
                # 融合CLIP特征
                if 'image_embeds' in clip_results and 'text_embeds' in clip_results:
                    # 使用注意力机制融合图像和文本特征
                    img_clip = clip_results['image_embeds']
                    text_clip = clip_results['text_embeds']
                    
                    # 简单的加权平均（可以改进为注意力机制）
                    clip_fused = (img_clip + text_clip) / 2
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
            # 投影所有特征到相同维度然后拼接
            standard_dim = 128
            projected_embeddings = []
            
            for i, emb in enumerate(valid_embeddings):
                proj_attr_name = f'fallback_projection_{i}'
                if not hasattr(self, proj_attr_name):
                    setattr(self, proj_attr_name, 
                           nn.Linear(emb.size(-1), standard_dim).to(emb.device))
                proj_layer = getattr(self, proj_attr_name)
                projected_embeddings.append(proj_layer(emb))
            
            # 拼接投影后的特征
            joint_emb = torch.cat(projected_embeddings, dim=-1)
            
            # 最终投影到目标维度
            if not hasattr(self, 'final_fallback_projection'):
                input_dim = joint_emb.size(-1)
                output_dim = getattr(self.args, 'joint_dim', 600)
                self.final_fallback_projection = nn.Linear(input_dim, output_dim).to(joint_emb.device)
            joint_emb = self.final_fallback_projection(joint_emb)
        else:
            # 如果没有任何有效特征，使用实体嵌入
            entity_input = self.entity_emb(input_idx)
            output_dim = getattr(self.args, 'joint_dim', 600)
            if not hasattr(self, 'emergency_projection'):
                self.emergency_projection = nn.Linear(entity_input.size(-1), output_dim).to(entity_input.device)
            joint_emb = self.emergency_projection(entity_input)
        
        return joint_emb


# 为了保持兼容性，设置别名
KAMultiModal = IBMultiModal
ClipEnhancedKACEA = IBMultiModal