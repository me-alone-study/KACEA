#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的多模态实体对齐模型 - 统一可配置架构
主要改进：
1. 模态可靠性评估 - 自动检测缺失/低质量模态
2. 自适应融合权重 - 实体级别的动态权重
3. 跨模态对齐机制 - 强制模态间一致性
4. 图像特征增强 - 更好的缺失处理策略
5. [新增] ResidualGCN - 残差图卷积网络
6. [新增] NeighborAggregator - 邻居信息聚合
7. [新增] 低质量模态过滤机制

=== 可配置的架构改进模块 ===
8. [DBP15K] CrossLingualEncoder - 跨语言名称编码
9. [DBP15K] AlignmentPropagation - 对齐传播网络
10. [DBP15K] MultiViewConsistency - 多视图一致性学习
11. [DBP15K] CharAwareAlignment - 字符级匹配
12. [FB15K] RelationalGCN - 关系感知图卷积
13. [FB15K] CrossKGAttention - 跨知识图谱注意力
14. [FB15K] GraphMatching - 图匹配网络
15. [通用] MomentumContrast - 动量对比学习
16. [通用] MultiGranularityAlignment - 多粒度对齐
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

# 多语言BERT导入（用于跨语言对齐）
try:
    from transformers import BertModel, BertTokenizer, XLMRobertaModel, XLMRobertaTokenizer
    MBERT_AVAILABLE = True
except ImportError:
    print("Warning: multilingual BERT not available")
    BertModel = BertTokenizer = XLMRobertaModel = XLMRobertaTokenizer = None
    MBERT_AVAILABLE = False

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


class ResidualGCN(nn.Module):
    """
    [新增] 带残差连接的GCN - 更深的网络结构
    改进点：
    1. 增加网络深度（4层）
    2. 添加残差连接防止梯度消失
    3. 批归一化+层归一化稳定训练
    4. Dropout增强正则化
    """
    def __init__(self, nfeat, nhid, nclass, dropout, num_layers=4):
        super(ResidualGCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 输入层
        self.gc_input = GraphConvolution(nfeat, nhid)
        self.bn_input = nn.BatchNorm1d(nhid)
        self.ln_input = nn.LayerNorm(nhid)
        
        # 中间层（带残差）
        self.gc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        for i in range(num_layers - 2):
            self.gc_layers.append(GraphConvolution(nhid, nhid))
            self.bn_layers.append(nn.BatchNorm1d(nhid))
            self.ln_layers.append(nn.LayerNorm(nhid))
        
        # 输出层
        self.gc_output = GraphConvolution(nhid, nclass)
        self.bn_output = nn.BatchNorm1d(nclass)
        self.ln_output = nn.LayerNorm(nclass)
        
        # 残差投影（如果维度不匹配）
        self.skip_connection = None
        if nfeat != nclass:
            self.skip_connection = nn.Linear(nfeat, nclass)

    def forward(self, x, adj):
        identity = x
        
        # 输入层
        x = self.gc_input(x, adj)
        x = self.bn_input(x) if x.size(0) > 1 else x
        x = self.ln_input(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 中间层（带残差）
        for gc, bn, ln in zip(self.gc_layers, self.bn_layers, self.ln_layers):
            residual = x
            x = gc(x, adj)
            x = bn(x) if x.size(0) > 1 else x  # BatchNorm需要batch_size > 1
            x = ln(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x + 0.8 * residual  # 强化残差连接
        
        # 输出层
        x = self.gc_output(x, adj)
        x = self.bn_output(x) if x.size(0) > 1 else x
        x = self.ln_output(x)
        
        # 全局残差连接
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        
        return x + 0.2 * identity  # 加权残差


class NeighborAggregator(nn.Module):
    """
    [新增] 邻居信息聚合模块
    用于增强实体表示，特别适合跨KG对齐任务
    """
    def __init__(self, input_dim, output_dim, aggregation='mean'):
        super(NeighborAggregator, self).__init__()
        self.aggregation = aggregation
        
        # 自身特征变换
        self.self_transform = nn.Linear(input_dim, output_dim)
        
        # 邻居特征变换
        self.neighbor_transform = nn.Linear(input_dim, output_dim)
        
        # 组合变换
        self.combine_transform = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # 注意力权重（可选）
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, 1),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x, adj):
        """
        Args:
            x: 实体特征 [N, input_dim]
            adj: 邻接矩阵 [N, N]
        Returns:
            enhanced: 增强后的特征 [N, output_dim]
        """
        # 自身特征
        self_feat = self.self_transform(x)
        
        # 聚合邻居特征
        if adj.is_sparse:
            neighbor_sum = torch.sparse.mm(adj, x)
            # 计算度数用于平均
            degree = torch.sparse.sum(adj, dim=1).to_dense().unsqueeze(1).clamp(min=1)
        else:
            neighbor_sum = torch.mm(adj, x)
            degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        
        neighbor_mean = neighbor_sum / degree
        neighbor_feat = self.neighbor_transform(neighbor_mean)
        
        # 组合自身和邻居特征
        combined = torch.cat([self_feat, neighbor_feat], dim=-1)
        enhanced = self.combine_transform(combined)
        
        return enhanced


# ============================================
# 新增架构改进模块（可配置）
# ============================================

class RelationalGCN(nn.Module):
    """
    [FB15K专用] 关系感知图卷积网络
    为不同关系类型使用不同的权重矩阵
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, num_bases=None, dropout=0.3):
        super(RelationalGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_bases = num_bases or min(num_relations, 100)
        self.dropout = dropout
        
        # 使用基分解减少参数量
        self.bases = nn.Parameter(torch.Tensor(self.num_bases, input_dim, hidden_dim))
        self.weights = nn.Parameter(torch.Tensor(num_relations, self.num_bases))
        
        # 自环权重
        self.self_loop_weight = nn.Linear(input_dim, hidden_dim)
        
        # 第二层
        self.bases2 = nn.Parameter(torch.Tensor(self.num_bases, hidden_dim, output_dim))
        self.weights2 = nn.Parameter(torch.Tensor(num_relations, self.num_bases))
        self.self_loop_weight2 = nn.Linear(hidden_dim, output_dim)
        
        # 归一化
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.bases2)
        nn.init.xavier_uniform_(self.weights2)
    
    def forward(self, x, adj, edge_type=None):
        """
        Args:
            x: [N, input_dim]
            adj: [N, N] 邻接矩阵
            edge_type: [N, N] 边的关系类型（可选）
        """
        # 第一层
        h = self._rgcn_layer(x, adj, edge_type, self.bases, self.weights, self.self_loop_weight)
        h = self.layer_norm1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        
        # 第二层
        h = self._rgcn_layer(h, adj, edge_type, self.bases2, self.weights2, self.self_loop_weight2)
        h = self.layer_norm2(h)
        
        return h
    
    def _rgcn_layer(self, x, adj, edge_type, bases, weights, self_loop):
        """单层R-GCN"""
        # 自环
        out = self_loop(x)
        
        # 如果没有边类型信息，使用标准GCN
        if edge_type is None:
            if adj.is_sparse:
                out = out + torch.sparse.mm(adj, x @ bases[0])
            else:
                out = out + torch.mm(adj, x @ bases[0])
        else:
            # 关系特定的传播
            for r in range(self.num_relations):
                # 获取关系r的权重矩阵（基分解）
                w_r = torch.sum(weights[r].unsqueeze(-1).unsqueeze(-1) * bases, dim=0)
                
                # 关系r的邻接矩阵
                adj_r = (edge_type == r).float() * adj
                
                # 传播
                if adj_r.is_sparse:
                    out = out + torch.sparse.mm(adj_r, x @ w_r)
                else:
                    out = out + torch.mm(adj_r, x @ w_r)
        
        return out


class CrossKGAttention(nn.Module):
    """
    [通用] 跨知识图谱注意力机制
    利用已知对齐在两个KG之间传递信息
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(CrossKGAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 多头注意力
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, kg1_emb, kg2_emb, alignment_matrix):
        """
        Args:
            kg1_emb: [N1, hidden_dim] KG1的实体嵌入
            kg2_emb: [N2, hidden_dim] KG2的实体嵌入
            alignment_matrix: [N1, N2] 对齐矩阵（0/1或概率）
        Returns:
            kg1_enhanced: [N1, hidden_dim]
            kg2_enhanced: [N2, hidden_dim]
        """
        batch_size_1 = kg1_emb.size(0)
        batch_size_2 = kg2_emb.size(0)
        
        # KG1查询KG2
        Q1 = self.query_proj(kg1_emb).view(batch_size_1, self.num_heads, self.head_dim)
        K2 = self.key_proj(kg2_emb).view(batch_size_2, self.num_heads, self.head_dim)
        V2 = self.value_proj(kg2_emb).view(batch_size_2, self.num_heads, self.head_dim)
        
        # 计算注意力分数
        attn_scores_1to2 = torch.einsum('nih,mjh->nmij', Q1, K2) * self.scale  # [N1, N2, num_heads, 1]
        attn_scores_1to2 = attn_scores_1to2.mean(dim=-1)  # [N1, N2, num_heads]
        
        # 用对齐矩阵作为mask
        alignment_mask = alignment_matrix.unsqueeze(-1).expand(-1, -1, self.num_heads)
        attn_scores_1to2 = attn_scores_1to2 * alignment_mask
        
        # Softmax
        attn_weights_1to2 = F.softmax(attn_scores_1to2, dim=1)
        attn_weights_1to2 = self.dropout(attn_weights_1to2)
        
        # 聚合
        kg1_from_kg2 = torch.einsum('nmh,mhd->nhd', attn_weights_1to2, V2)
        kg1_from_kg2 = kg1_from_kg2.reshape(batch_size_1, self.hidden_dim)
        kg1_enhanced = kg1_emb + self.out_proj(kg1_from_kg2)
        
        # KG2查询KG1（对称操作）
        Q2 = self.query_proj(kg2_emb).view(batch_size_2, self.num_heads, self.head_dim)
        K1 = self.key_proj(kg1_emb).view(batch_size_1, self.num_heads, self.head_dim)
        V1 = self.value_proj(kg1_emb).view(batch_size_1, self.num_heads, self.head_dim)
        
        attn_scores_2to1 = torch.einsum('mih,njh->mnij', Q2, K1) * self.scale
        attn_scores_2to1 = attn_scores_2to1.mean(dim=-1)
        
        alignment_mask_t = alignment_matrix.t().unsqueeze(-1).expand(-1, -1, self.num_heads)
        attn_scores_2to1 = attn_scores_2to1 * alignment_mask_t
        
        attn_weights_2to1 = F.softmax(attn_scores_2to1, dim=1)
        attn_weights_2to1 = self.dropout(attn_weights_2to1)
        
        kg2_from_kg1 = torch.einsum('mnh,nhd->mhd', attn_weights_2to1, V1)
        kg2_from_kg1 = kg2_from_kg1.reshape(batch_size_2, self.hidden_dim)
        kg2_enhanced = kg2_emb + self.out_proj(kg2_from_kg1)
        
        return kg1_enhanced, kg2_enhanced


class AlignmentPropagationNetwork(nn.Module):
    """
    [DBP15K专用] 对齐传播网络
    在跨KG图上传播对齐信息
    """
    def __init__(self, hidden_dim, num_layers=2, dropout=0.1):
        super(AlignmentPropagationNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 多层跨KG注意力
        self.cross_kg_layers = nn.ModuleList([
            CrossKGAttention(hidden_dim, num_heads=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 门控机制
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(num_layers)
        ])
        
    def forward(self, kg1_emb, kg2_emb, train_links):
        """
        Args:
            kg1_emb: [N1, hidden_dim]
            kg2_emb: [N2, hidden_dim]
            train_links: [num_links, 2] 训练对齐对
        Returns:
            kg1_prop: [N1, hidden_dim] 传播后的KG1嵌入
            kg2_prop: [N2, hidden_dim] 传播后的KG2嵌入
        """
        # 构建对齐矩阵
        alignment_matrix = self._build_alignment_matrix(
            kg1_emb.size(0), kg2_emb.size(0), train_links, kg1_emb.device
        )
        
        kg1_prop = kg1_emb
        kg2_prop = kg2_emb
        
        # 多层传播
        for layer_idx in range(self.num_layers):
            # 跨KG注意力
            kg1_new, kg2_new = self.cross_kg_layers[layer_idx](
                kg1_prop, kg2_prop, alignment_matrix
            )
            
            # 门控融合
            gate1 = self.gates[layer_idx](torch.cat([kg1_prop, kg1_new], dim=-1))
            kg1_prop = gate1 * kg1_new + (1 - gate1) * kg1_prop
            
            gate2 = self.gates[layer_idx](torch.cat([kg2_prop, kg2_new], dim=-1))
            kg2_prop = gate2 * kg2_new + (1 - gate2) * kg2_prop
        
        return kg1_prop, kg2_prop
    
    def _build_alignment_matrix(self, n1, n2, train_links, device):
        """构建对齐矩阵"""
        alignment_matrix = torch.zeros(n1, n2, device=device)
        if len(train_links) > 0:
            valid_links = train_links[
                (train_links[:, 0] < n1) & (train_links[:, 1] < n2)
            ]
            if len(valid_links) > 0:
                alignment_matrix[valid_links[:, 0], valid_links[:, 1]] = 1.0
        return alignment_matrix


class CrossLingualNameEncoder(nn.Module):
    """
    [DBP15K专用] 跨语言名称编码器
    使用多语言BERT处理中英文实体名称
    """
    def __init__(self, model_name='bert-base-multilingual-cased', output_dim=128, freeze=True):
        super(CrossLingualNameEncoder, self).__init__()
        
        if not MBERT_AVAILABLE:
            raise ImportError("transformers library required for multilingual BERT")
        
        # 使用多语言BERT或XLM-RoBERTa
        if 'xlm-roberta' in model_name.lower():
            self.encoder = XLMRobertaModel.from_pretrained(model_name)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        else:
            self.encoder = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, entity_names, batch_size=64):
        """
        Args:
            entity_names: List[str] 实体名称列表
            batch_size: 批处理大小
        Returns:
            embeddings: [N, output_dim]
        """
        all_embeddings = []
        
        for i in range(0, len(entity_names), batch_size):
            batch_names = entity_names[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_names,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )
            inputs = {k: v.to(next(self.encoder.parameters()).device) for k, v in inputs.items()}
            
            # Encode
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.encoder(**inputs)
                # 使用[CLS] token的表示
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Project
            projected = self.projection(cls_embeddings)
            all_embeddings.append(projected)
        
        return torch.cat(all_embeddings, dim=0)


class GraphMatchingNetwork(nn.Module):
    """
    [FB15K专用] 图匹配网络
    考虑实体邻域结构的相似性
    """
    def __init__(self, hidden_dim, k_hop=2):
        super(GraphMatchingNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.k_hop = k_hop
        
        # 邻域编码器
        self.neighborhood_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 图匹配得分网络
        self.matching_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, emb, adj, entity_pairs):
        """
        Args:
            emb: [N, hidden_dim] 实体嵌入
            adj: [N, N] 邻接矩阵
            entity_pairs: [num_pairs, 2] 待匹配的实体对
        Returns:
            matching_scores: [num_pairs] 匹配分数
        """
        # 提取k-hop邻域
        neighborhoods = self._extract_neighborhoods(emb, adj, self.k_hop)
        
        # 对每个实体对计算匹配分数
        e1_indices = entity_pairs[:, 0]
        e2_indices = entity_pairs[:, 1]
        
        e1_emb = emb[e1_indices]
        e2_emb = emb[e2_indices]
        e1_neigh = neighborhoods[e1_indices]
        e2_neigh = neighborhoods[e2_indices]
        
        # 拼接特征：[实体1, 实体2, 邻域1, 邻域2]
        combined = torch.cat([e1_emb, e2_emb, e1_neigh, e2_neigh], dim=-1)
        
        # 计算匹配分数
        matching_scores = self.matching_network(combined).squeeze(-1)
        
        return matching_scores
    
    def _extract_neighborhoods(self, emb, adj, k_hop):
        """提取k-hop邻域表示"""
        # 简化版：使用k次矩阵乘法聚合k-hop邻居
        neighborhood_emb = emb
        adj_power = adj
        
        for _ in range(k_hop):
            if adj_power.is_sparse:
                neighborhood_emb = torch.sparse.mm(adj_power, neighborhood_emb)
            else:
                neighborhood_emb = torch.mm(adj_power, neighborhood_emb)
        
        # 编码
        return self.neighborhood_encoder(neighborhood_emb)


class MultiViewConsistencyModule(nn.Module):
    """
    [DBP15K专用] 多视图一致性模块
    确保不同模态的对齐决策一致
    """
    def __init__(self, view_names):
        super(MultiViewConsistencyModule, self).__init__()
        self.view_names = view_names
    
    def compute_consistency_loss(self, embeddings_dict, links, temperature=0.1):
        """
        计算多视图一致性损失
        Args:
            embeddings_dict: Dict[str, Tensor] 各视图的嵌入
            links: [num_links, 2] 对齐对
            temperature: 温度参数
        Returns:
            consistency_loss: 标量损失
        """
        if len(embeddings_dict) < 2:
            return torch.tensor(0.0, device=next(iter(embeddings_dict.values())).device)
        
        # 计算每个视图的相似度矩阵
        similarities = {}
        for view_name, emb in embeddings_dict.items():
            if emb is not None and len(links) > 0:
                e1 = F.normalize(emb[links[:, 0]], dim=-1)
                e2 = F.normalize(emb[links[:, 1]], dim=-1)
                sim = torch.sum(e1 * e2, dim=-1) / temperature
                similarities[view_name] = sim
        
        if len(similarities) < 2:
            return torch.tensor(0.0, device=next(iter(embeddings_dict.values())).device)
        
        # 计算视图间的一致性损失
        consistency_loss = 0.0
        view_list = list(similarities.keys())
        count = 0
        
        for i in range(len(view_list)):
            for j in range(i+1, len(view_list)):
                # KL散度或MSE
                sim_i = F.softmax(similarities[view_list[i]], dim=0)
                sim_j = F.softmax(similarities[view_list[j]], dim=0)
                
                # 使用KL散度
                kl_loss = F.kl_div(
                    torch.log(sim_i + 1e-10),
                    sim_j,
                    reduction='batchmean'
                )
                consistency_loss += kl_loss
                count += 1
        
        return consistency_loss / max(count, 1)


class MomentumContrastEncoder(nn.Module):
    """
    [通用] 动量对比编码器
    使用momentum更新的队列进行对比学习
    """
    def __init__(self, encoder, dim=128, K=4096, m=0.999, T=0.07):
        super(MomentumContrastEncoder, self).__init__()
        
        self.K = K  # 队列大小
        self.m = m  # momentum系数
        self.T = T  # 温度
        
        # Query encoder
        self.encoder_q = encoder
        
        # Key encoder (momentum)
        self.encoder_k = type(encoder)(encoder.args, encoder.ENT_NUM, 
                                       encoder.img_feature_dim if hasattr(encoder, 'img_feature_dim') else 2048,
                                       encoder.char_feature_dim if hasattr(encoder, 'char_feature_dim') else 100,
                                       encoder.use_project_head if hasattr(encoder, 'use_project_head') else False)
        
        # 初始化key encoder与query encoder相同
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # 创建队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum更新key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的键
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 分两部分更新
            self.queue[:, ptr:] = keys[:self.K - ptr].T
            self.queue[:, :(ptr + batch_size) % self.K] = keys[self.K - ptr:].T
        
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    def forward(self, x_q, x_k):
        """
        Args:
            x_q: query样本
            x_k: key样本（正样本）
        Returns:
            logits, labels
        """
        # Query特征
        q = self.encoder_q(x_q)[-1]  # 取joint_emb
        q = F.normalize(q, dim=-1)
        
        # Key特征（不计算梯度）
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(x_k)[-1]
            k = F.normalize(k, dim=-1)
        
        # 正样本logit
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # 负样本logits（从队列）
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # 拼接
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        
        # 标签（正样本在第0位）
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # 更新队列
        self._dequeue_and_enqueue(k)
        
        return logits, labels


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
                x = layer(x, adj)
                
                if x.dim() == 3 and x.size(0) == n_head:
                    x = x.mean(dim=0)
                
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
    """CLIP实体编码器"""
    
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
        self.feature_aligner = None
    
    def encode_images_from_raw(self, images):
        """从原始图像编码"""
        try:
            if isinstance(images, list):
                inputs = self.clip_processor(images=images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.device)
            else:
                pixel_values = images.to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
            
            return F.normalize(image_features, dim=-1)
        
        except Exception as e:
            print(f"Warning: Failed to encode raw images with ViT: {e}")
            return None
    
    def encode_images(self, image_features, use_alignment=True):
        """编码预提取的图像特征"""
        input_dim = image_features.size(-1)
        
        if use_alignment:
            if self.feature_aligner is None or self.feature_aligner[0].in_features != input_dim:
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
        """编码文本"""
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
    
    def forward(self, image_features=None, texts=None, raw_images=None):
        """前向传播"""
        results = {}
        
        if raw_images is not None:
            vit_features = self.encode_images_from_raw(raw_images)
            if vit_features is not None:
                results['image_embeds'] = vit_features
                results['used_vit'] = True
        
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
        
        self.reliability_nets = nn.ModuleDict()
        for modal_name, modal_dim in modal_dims.items():
            self.reliability_nets[modal_name] = nn.Sequential(
                nn.Linear(modal_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        self.modal_means = {}
        self.modal_stds = {}
    
    def compute_feature_quality(self, features, modal_name):
        """计算特征质量分数"""
        if features is None:
            return torch.zeros(1, device=next(self.parameters()).device)
        
        feature_var = features.var(dim=-1, keepdim=True)
        feature_norm = features.norm(dim=-1, keepdim=True)
        
        features_normalized = F.normalize(features, dim=-1)
        
        if modal_name in self.reliability_nets:
            reliability = self.reliability_nets[modal_name](features)
        else:
            reliability = torch.ones(features.size(0), 1, device=features.device)
        
        var_score = torch.sigmoid(feature_var * 10 - 0.5)
        norm_score = torch.sigmoid(feature_norm - 0.5)
        
        quality_score = (reliability + var_score + norm_score) / 3.0
        
        return quality_score.squeeze(-1)
    
    def forward(self, modal_features):
        """计算所有模态的可靠性分数"""
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
    自适应多模态融合层 - 改进版
    改进点：
    1. 实体级别的动态权重
    2. 基于可靠性的加权
    3. 跨模态注意力交互
    4. [新增] 低质量模态过滤
    """
    
    def __init__(self, modal_dims, output_dim, use_cross_modal_attention=True):
        super(AdaptiveMultiModalFusion, self).__init__()
        self.modal_dims = modal_dims
        self.output_dim = output_dim
        self.use_cross_modal_attention = use_cross_modal_attention
        
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
        
        # 自适应门控网络
        num_modals = len(modal_dims)
        self.gate_network = nn.Sequential(
            nn.Linear(num_modals * self.standard_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_modals),
            nn.Softmax(dim=-1)
        )
        
        # 跨模态注意力
        if use_cross_modal_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.standard_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
        
        # 最终投影 - 动态计算输入维度
        self.num_modals = num_modals
        self.output_projection = None  # 延迟初始化
        
        # 残差连接的投影
        self.residual_projections = nn.ModuleDict()
        for modal_name in modal_dims.keys():
            self.residual_projections[modal_name] = nn.Linear(self.standard_dim, output_dim)
    
    def _filter_low_quality_modals(self, modal_features, threshold=0.3):
        """
        [新增] 过滤低质量模态
        对于大量缺失（非零比例低）的模态，直接跳过
        """
        filtered = {}
        for name, feat in modal_features.items():
            if feat is not None:
                # 检查非零比例
                non_zero_ratio = (feat.abs().sum(dim=-1) > 1e-6).float().mean().item()
                if non_zero_ratio > threshold:
                    filtered[name] = feat
                else:
                    # 低质量模态，跳过
                    pass
        return filtered
    
    def forward(self, modal_features, knowledge_context=None):
        """自适应融合多模态特征"""
        device = next(self.parameters()).device
        
        # [新增] 过滤低质量模态
        modal_features = self._filter_low_quality_modals(modal_features, threshold=0.3)
        
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
        
        # 3. 跨模态注意力
        if self.use_cross_modal_attention and len(valid_modals) > 1:
            modal_stack = torch.stack([projected_features[m] for m in valid_modals], dim=1)
            attended, _ = self.cross_attention(modal_stack, modal_stack, modal_stack)
            for i, modal_name in enumerate(valid_modals):
                projected_features[modal_name] = attended[:, i, :]
        
        # 4. 计算自适应门控权重
        concat_features = []
        for modal_name in self.modal_dims.keys():
            if modal_name in projected_features:
                concat_features.append(projected_features[modal_name])
            else:
                concat_features.append(torch.zeros(batch_size, self.standard_dim, device=device))
        
        concat_features = torch.cat(concat_features, dim=-1)
        gate_weights = self.gate_network(concat_features)
        
        # 5. 结合可靠性分数调整权重
        adjusted_weights = []
        for i, modal_name in enumerate(self.modal_dims.keys()):
            if modal_name in reliability_scores and reliability_scores[modal_name] is not None:
                rel_score = reliability_scores[modal_name][:batch_size]
                adjusted_weight = gate_weights[:, i] * rel_score
            else:
                adjusted_weight = gate_weights[:, i] * 0.1
            adjusted_weights.append(adjusted_weight)
        
        adjusted_weights = torch.stack(adjusted_weights, dim=-1)
        adjusted_weights = F.softmax(adjusted_weights, dim=-1)
        
        # 6. 加权融合
        weighted_features = []
        for i, modal_name in enumerate(self.modal_dims.keys()):
            if modal_name in projected_features:
                weighted = projected_features[modal_name] * adjusted_weights[:, i:i+1]
            else:
                weighted = torch.zeros(batch_size, self.standard_dim, device=device)
            weighted_features.append(weighted)
        
        fused = torch.cat(weighted_features, dim=-1)
        
        # 7. 最终投影 - 动态初始化以适配实际输入维度
        if self.output_projection is None or self.output_projection[0].in_features != fused.size(-1):
            self.output_projection = nn.Sequential(
                nn.Linear(fused.size(-1), self.output_dim),
                nn.LayerNorm(self.output_dim)
            ).to(fused.device)
        
        output = self.output_projection(fused)
        
        # 8. 残差连接
        if knowledge_context is not None and 'graph' in projected_features:
            residual = self.residual_projections['graph'](projected_features['graph'])
            output = output + 0.1 * residual
        
        return output


class CrossModalAlignmentModule(nn.Module):
    """跨模态对齐模块"""
    
    def __init__(self, modal_dims, align_dim=128):
        super(CrossModalAlignmentModule, self).__init__()
        self.modal_dims = modal_dims
        self.align_dim = align_dim
        
        self.align_projections = nn.ModuleDict()
        for modal_name, modal_dim in modal_dims.items():
            self.align_projections[modal_name] = nn.Sequential(
                nn.Linear(modal_dim, align_dim),
                nn.LayerNorm(align_dim)
            )
    
    def compute_alignment_loss(self, modal_features, train_links):
        """计算跨模态对齐损失"""
        device = next(self.parameters()).device
        
        valid_modals = {k: v for k, v in modal_features.items() if v is not None}
        
        if len(valid_modals) < 2:
            return torch.tensor(0.0, device=device)
        
        total_loss = 0.0
        pair_count = 0
        
        modal_names = list(valid_modals.keys())
        
        for i in range(len(modal_names)):
            for j in range(i + 1, len(modal_names)):
                modal_i = modal_names[i]
                modal_j = modal_names[j]
                
                feat_i = valid_modals[modal_i]
                feat_j = valid_modals[modal_j]
                
                proj_i = self.align_projections[modal_i](feat_i)
                proj_j = self.align_projections[modal_j](feat_j)
                
                proj_i = F.normalize(proj_i, dim=-1)
                proj_j = F.normalize(proj_j, dim=-1)
                
                min_size = min(proj_i.size(0), proj_j.size(0))
                valid_links = train_links[train_links[:, 0] < min_size]
                valid_links = valid_links[valid_links[:, 1] < min_size]
                
                if len(valid_links) == 0:
                    continue
                
                aligned_i = proj_i[valid_links[:, 0]]
                aligned_j = proj_j[valid_links[:, 1]]
                
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
    4. [新增] 可选ResidualGCN
    5. [新增] 邻居聚合增强
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

        # [改进] 关系特征处理层 - 更深的网络，支持增强关系特征
        # 自动检测输入维度(基础1000 或 增强2004)
        self.rel_input_dim = 1000  # 将在forward中动态调整
        
        # 使用可变输入维度的深层网络
        self.rel_fc_input = None  # 延迟初始化
        self.rel_fc_core = nn.Sequential(
            nn.Linear(attr_dim * 8, attr_dim * 6),
            nn.BatchNorm1d(attr_dim * 6),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(attr_dim * 6, attr_dim * 4),
            nn.BatchNorm1d(attr_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(attr_dim * 4, attr_dim * 2),
            nn.BatchNorm1d(attr_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attr_dim * 2, attr_dim)
        )
        # 残差投影
        self.rel_skip = None  # 延迟初始化
        
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

        # [改进] 图结构编码器 - 支持ResidualGCN
        no_diag = getattr(args, 'no_diag', False)
        diag = not no_diag
        self._init_structure_encoder(dropout, diag)
        
        # [新增] 邻居聚合模块
        self.use_neighbor_agg = getattr(args, 'use_neighbor_agg', False)
        if self.use_neighbor_agg:
            self.neighbor_aggregator = NeighborAggregator(
                self.n_units[-1], self.n_units[-1]
            )
        
        # ============================================
        # [新增] 可配置的架构改进模块
        # ============================================
        
        # [FB15K] R-GCN - 关系感知图卷积
        self.use_rgcn = getattr(args, 'use_rgcn', False)
        self.rgcn_encoder = None
        if self.use_rgcn:
            num_relations = getattr(args, 'num_relations', 100)
            self.rgcn_encoder = RelationalGCN(
                self.input_dim, self.n_units[1] if len(self.n_units) > 1 else self.input_dim,
                self.n_units[-1], num_relations, dropout=dropout
            )
        
        # [通用] 跨KG注意力
        self.use_cross_kg_attention = getattr(args, 'use_cross_kg_attention', False)
        self.cross_kg_attention = None
        if self.use_cross_kg_attention:
            self.cross_kg_attention = CrossKGAttention(
                self.n_units[-1], num_heads=4, dropout=dropout
            )
        
        # [DBP15K] 对齐传播网络
        self.use_alignment_propagation = getattr(args, 'use_alignment_propagation', False)
        self.alignment_propagation = None
        if self.use_alignment_propagation:
            self.alignment_propagation = AlignmentPropagationNetwork(
                self.n_units[-1], num_layers=2, dropout=dropout
            )
        
        # [DBP15K] 跨语言名称编码器
        self.use_cross_lingual_encoder = getattr(args, 'use_cross_lingual_encoder', False)
        self.cross_lingual_encoder = None
        if self.use_cross_lingual_encoder and MBERT_AVAILABLE:
            try:
                mbert_model = getattr(args, 'mbert_model', 'bert-base-multilingual-cased')
                self.cross_lingual_encoder = CrossLingualNameEncoder(
                    model_name=mbert_model,
                    output_dim=char_dim,
                    freeze=getattr(args, 'mbert_freeze', True)
                )
                print(f"Cross-lingual encoder initialized: {mbert_model}")
            except Exception as e:
                print(f"Warning: Failed to initialize cross-lingual encoder: {e}")
                self.use_cross_lingual_encoder = False
        
        # [FB15K] 图匹配网络
        self.use_graph_matching = getattr(args, 'use_graph_matching', False)
        self.graph_matching = None
        if self.use_graph_matching:
            self.graph_matching = GraphMatchingNetwork(
                self.n_units[-1], k_hop=getattr(args, 'matching_k_hop', 2)
            )
        
        # [DBP15K] 多视图一致性
        self.use_multi_view_consistency = getattr(args, 'use_multi_view_consistency', False)
        self.multi_view_consistency = None
        if self.use_multi_view_consistency:
            view_names = []
            if args.w_gcn: view_names.append('graph')
            if args.w_img: view_names.append('image')
            if args.w_rel: view_names.append('relation')
            if args.w_attr: view_names.append('attribute')
            if args.w_name: view_names.append('name')
            if args.w_char: view_names.append('char')
            
            self.multi_view_consistency = MultiViewConsistencyModule(view_names)

        # 多模态融合
        modal_dims = {}
        if args.w_gcn: modal_dims['graph'] = self.n_units[-1]
        if args.w_img: modal_dims['image'] = img_dim
        if args.w_rel: modal_dims['relation'] = attr_dim  
        if args.w_attr: modal_dims['attribute'] = attr_dim
        if args.w_name: modal_dims['name'] = char_dim
        if args.w_char: modal_dims['char'] = char_dim
        if self.use_clip: modal_dims['clip'] = img_dim
        
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
        
        # 模态特征记录
        self.modal_features = {}

    def _init_structure_encoder(self, dropout, diag):
        """初始化图结构编码器"""
        structure_encoder = getattr(self.args, 'structure_encoder', 'gcn')
        use_residual_gcn = getattr(self.args, 'use_residual_gcn', False)
        
        if structure_encoder == 'gcn':
            if use_residual_gcn:
                # [新增] 使用ResidualGCN
                if self.use_graph_vib:
                    self.cross_graph_model_mu = ResidualGCN(
                        self.input_dim, 
                        self.n_units[1] if len(self.n_units) > 1 else self.input_dim,
                        self.n_units[-1], 
                        dropout,
                        num_layers=len(self.n_units)
                    )
                    self.cross_graph_model_logvar = ResidualGCN(
                        self.input_dim, 
                        self.n_units[1] if len(self.n_units) > 1 else self.input_dim,
                        self.n_units[-1], 
                        dropout,
                        num_layers=len(self.n_units)
                    )
                else:
                    self.cross_graph_model = ResidualGCN(
                        self.input_dim, 
                        self.n_units[1] if len(self.n_units) > 1 else self.input_dim,
                        self.n_units[-1], 
                        dropout,
                        num_layers=len(self.n_units)
                    )
            else:
                # 原始SimpleGCN
                if self.use_graph_vib:
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
        """前向传播"""
        modal_features = {}
        
        # === 图结构特征处理 ===
        gph_emb = None
        if self.args.w_gcn:
            try:
                entity_input = self.entity_emb(input_idx)
                
                # [新增] 优先使用R-GCN（如果启用）
                if self.use_rgcn and self.rgcn_encoder is not None:
                    edge_type = rel_features if rel_features is not None else None
                    gph_emb = self.rgcn_encoder(entity_input, adj, edge_type)
                elif self.use_graph_vib and hasattr(self, 'cross_graph_model_mu'):
                    gph_mu = self.cross_graph_model_mu(entity_input, adj)
                    gph_logvar = self.cross_graph_model_logvar(entity_input, adj)
                    gph_logvar = torch.clamp(gph_logvar, -10, 2)
                    gph_std = torch.exp(0.5 * gph_logvar)
                    eps = torch.randn_like(gph_std)
                    gph_emb = gph_mu + eps * gph_std
                    self.kld_loss = self._kld_gauss(
                        gph_mu, gph_std, 
                        torch.zeros_like(gph_mu), torch.ones_like(gph_std)
                    )
                else:
                    gph_emb = self.cross_graph_model(entity_input, adj)
                
                # [新增] 邻居聚合增强
                if self.use_neighbor_agg and hasattr(self, 'neighbor_aggregator'):
                    gph_emb = gph_emb + 0.1 * self.neighbor_aggregator(gph_emb, adj)
                    
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
                # 动态初始化输入层(支持1000维或2004维)
                rel_input_dim = rel_features.size(-1)
                
                # 确保batch normalization的维度正确
                if rel_features.size(0) == 1:
                    # 如果batch size为1，跳过batch norm
                    if self.rel_fc_input is None or self.rel_fc_input[0].in_features != rel_input_dim:
                        self.rel_fc_input = nn.Sequential(
                            nn.Linear(rel_input_dim, self.args.attr_dim * 8),
                            nn.LayerNorm(self.args.attr_dim * 8),
                            nn.ReLU(),
                            nn.Dropout(self.args.dropout * 0.5)
                        ).to(rel_features.device)
                        # 残差投影
                        self.rel_skip = nn.Linear(rel_input_dim, self.args.attr_dim).to(rel_features.device)
                else:
                    if self.rel_fc_input is None or self.rel_fc_input[0].in_features != rel_input_dim:
                        self.rel_fc_input = nn.Sequential(
                            nn.Linear(rel_input_dim, self.args.attr_dim * 8),
                            nn.BatchNorm1d(self.args.attr_dim * 8),
                            nn.ReLU(),
                            nn.Dropout(self.args.dropout * 0.5)
                        ).to(rel_features.device)
                        # 残差投影
                        self.rel_skip = nn.Linear(rel_input_dim, self.args.attr_dim).to(rel_features.device)
                
                # 前向传播
                rel_h = self.rel_fc_input(rel_features)
                
                # 确保rel_fc_core的batch norm也能处理
                if rel_features.size(0) > 1:
                    rel_emb = self.rel_fc_core(rel_h)
                else:
                    # batch size为1时，跳过batch norm层
                    rel_emb = rel_h
                    for layer in self.rel_fc_core:
                        if not isinstance(layer, nn.BatchNorm1d):
                            rel_emb = layer(rel_emb)
                
                # 残差连接
                if self.rel_skip is not None:
                    rel_emb = rel_emb + 0.1 * self.rel_skip(rel_features)
                
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
                import traceback
                traceback.print_exc()

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
                # [新增] 优先使用跨语言编码器（如果启用且提供了entity_texts）
                if self.use_cross_lingual_encoder and self.cross_lingual_encoder is not None and entity_texts is not None:
                    name_emb = self.cross_lingual_encoder(entity_texts)
                else:
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

        # 保存模态特征
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
        
        # ============================================
        # [新增] 对齐传播和跨KG注意力（如果启用）
        # ============================================
        # 注意：由于维度匹配问题，暂时禁用自动调用
        # 需要在训练循环中手动调用或确保维度正确后再启用
        # if hasattr(self, '_apply_cross_kg_modules'):
        #     joint_emb, gph_emb = self._apply_cross_kg_modules(
        #         joint_emb, gph_emb, modal_features
        #     )

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
    
    def set_kg_split(self, left_ents, right_ents):
        """设置KG划分信息（用于跨KG模块）"""
        self.left_ents = left_ents
        self.right_ents = right_ents
    
    def _apply_cross_kg_modules(self, joint_emb, gph_emb, modal_features):
        """
        应用跨KG模块（对齐传播、跨KG注意力）
        需要先调用set_kg_split设置KG划分
        """
        if not hasattr(self, 'left_ents') or not hasattr(self, 'right_ents'):
            return joint_emb, gph_emb
        
        if not hasattr(self, 'train_links'):
            return joint_emb, gph_emb
        
        try:
            # 检查维度是否匹配
            if joint_emb is None:
                return joint_emb, gph_emb
            
            # 确保索引在范围内
            max_left = max(self.left_ents) if len(self.left_ents) > 0 else 0
            max_right = max(self.right_ents) if len(self.right_ents) > 0 else 0
            
            if max_left >= joint_emb.size(0) or max_right >= joint_emb.size(0):
                print(f"Warning: Index out of range in cross-KG modules, skipping")
                return joint_emb, gph_emb
            
            # 分离两个KG的嵌入
            kg1_emb = joint_emb[self.left_ents]
            kg2_emb = joint_emb[self.right_ents]
            
            # 检查维度
            if kg1_emb.size(-1) != kg2_emb.size(-1):
                print(f"Warning: Dimension mismatch in cross-KG modules: {kg1_emb.size(-1)} vs {kg2_emb.size(-1)}, skipping")
                return joint_emb, gph_emb
            
            # [新增] 对齐传播
            if self.use_alignment_propagation and self.alignment_propagation is not None:
                # 检查模块的hidden_dim是否匹配
                if self.alignment_propagation.hidden_dim == kg1_emb.size(-1):
                    kg1_prop, kg2_prop = self.alignment_propagation(
                        kg1_emb, kg2_emb, self.train_links
                    )
                    joint_emb[self.left_ents] = kg1_prop
                    joint_emb[self.right_ents] = kg2_prop
                else:
                    print(f"Warning: AlignmentPropagation dim mismatch: {self.alignment_propagation.hidden_dim} vs {kg1_emb.size(-1)}, skipping")
            
            # [新增] 跨KG注意力
            if self.use_cross_kg_attention and self.cross_kg_attention is not None:
                # 检查模块的hidden_dim是否匹配
                if self.cross_kg_attention.hidden_dim == kg1_emb.size(-1):
                    alignment_matrix = self._build_alignment_matrix(
                        len(self.left_ents), len(self.right_ents), self.train_links
                    )
                    kg1_attn, kg2_attn = self.cross_kg_attention(
                        kg1_emb, kg2_emb, alignment_matrix
                    )
                    # 融合原始特征和注意力增强特征
                    joint_emb[self.left_ents] = 0.7 * joint_emb[self.left_ents] + 0.3 * kg1_attn
                    joint_emb[self.right_ents] = 0.7 * joint_emb[self.right_ents] + 0.3 * kg2_attn
                else:
                    print(f"Warning: CrossKGAttention dim mismatch: {self.cross_kg_attention.hidden_dim} vs {kg1_emb.size(-1)}, skipping")
        
        except Exception as e:
            print(f"Warning: Cross-KG modules failed: {e}")
        
        return joint_emb, gph_emb
    
    def _build_alignment_matrix(self, n1, n2, train_links):
        """构建对齐矩阵（辅助函数）"""
        device = next(self.parameters()).device
        alignment_matrix = torch.zeros(n1, n2, device=device)
        if len(train_links) > 0:
            valid_links = train_links[
                (train_links[:, 0] < n1) & (train_links[:, 1] < n2)
            ]
            if len(valid_links) > 0:
                # 将全局索引转换为局部索引
                left_to_local = {ent: i for i, ent in enumerate(self.left_ents)}
                right_to_local = {ent: i for i, ent in enumerate(self.right_ents)}
                
                for link in valid_links:
                    if link[0] in left_to_local and link[1] in right_to_local:
                        i = left_to_local[link[0]]
                        j = right_to_local[link[1]]
                        alignment_matrix[i, j] = 1.0
        return alignment_matrix
    
    def get_multi_view_consistency_loss(self, train_links):
        """
        获取多视图一致性损失
        确保不同模态对相同实体对的对齐决策一致
        """
        if not self.use_multi_view_consistency or self.multi_view_consistency is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        if not hasattr(self, 'modal_features') or len(self.modal_features) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # 只使用非None的模态
        valid_modal_features = {k: v for k, v in self.modal_features.items() if v is not None}
        
        if len(valid_modal_features) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        try:
            consistency_loss = self.multi_view_consistency.compute_consistency_loss(
                valid_modal_features, train_links, temperature=0.1
            )
            return consistency_loss
        except Exception as e:
            print(f"Warning: Multi-view consistency loss failed: {e}")
            return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_graph_matching_loss(self, adj, entity_pairs, labels):
        """
        获取图匹配损失
        Args:
            adj: 邻接矩阵
            entity_pairs: [num_pairs, 2] 实体对
            labels: [num_pairs] 0/1标签，1表示对齐
        Returns:
            matching_loss: 标量损失
        """
        if not self.use_graph_matching or self.graph_matching is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        if not hasattr(self, 'modal_features') or 'graph' not in self.modal_features:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        try:
            gph_emb = self.modal_features['graph']
            matching_scores = self.graph_matching(gph_emb, adj, entity_pairs)
            
            # 二元交叉熵损失
            matching_loss = F.binary_cross_entropy(
                matching_scores, labels.float()
            )
            return matching_loss
        except Exception as e:
            print(f"Warning: Graph matching loss failed: {e}")
            return torch.tensor(0.0, device=next(self.parameters()).device)


# 兼容性别名
KAMultiModal = IBMultiModal
ClipEnhancedKACEA = IBMultiModal
KACEA = IBMultiModal
