#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的数据加载模块 - 增强版
主要改进：
1. 图像特征缺失的智能处理
2. 图像质量标记
3. 跨KG图像对齐预处理
4. [新增] 增强版关系特征 - 区分入度/出度、头/尾实体角色
"""

import numpy as np
import pickle
import json
import os
from collections import Counter, defaultdict
from tqdm import tqdm
import torch


def loadfile(fn, num=1):
    """加载文件，返回元组列表"""
    print("loading a file..." + fn)
    ret = []
    try:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                x = []
                for i in range(num):
                    x.append(int(th[i]))
                ret.append(tuple(x))
    except Exception as e:
        print(f"Warning: Failed to load file {fn}: {e}")
    return ret


def get_ids(fn):
    """从文件中获取ID列表"""
    ids = []
    try:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                ids.append(int(th[0]))
    except Exception as e:
        print(f"Warning: Failed to get IDs from {fn}: {e}")
    return ids


def get_ent2id(fns):
    """构建实体到ID的映射"""
    ent2id = {}
    for fn in fns:
        try:
            with open(fn, "r", encoding="utf-8") as f:
                for line in f:
                    th = line[:-1].split("\t")
                    if len(th) >= 2:
                        ent2id[th[1]] = int(th[0])
        except Exception as e:
            print(f"Warning: Failed to load entity mapping from {fn}: {e}")
    return ent2id


def load_attr(fns, e, ent2id, topA=1000):
    """加载属性特征"""
    cnt = {}
    for fn in fns:
        if not os.path.exists(fn):
            print(f"Warning: Attribute file {fn} not found, skipping...")
            continue
        try:
            with open(fn, "r", encoding="utf-8") as f:
                for line in f:
                    th = line[:-1].split("\t")
                    if len(th) < 2 or th[0] not in ent2id:
                        continue
                    for i in range(1, len(th)):
                        if th[i] not in cnt:
                            cnt[th[i]] = 1
                        else:
                            cnt[th[i]] += 1
        except Exception as e:
            print(f"Warning: Failed to process attribute file {fn}: {e}")
    
    if not cnt:
        print(f"Warning: No attributes found, creating random features")
        return np.random.normal(0, 0.1, (e, topA)).astype(np.float32)
    
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    for i in range(min(topA, len(fre))):
        attr2id[fre[i][0]] = i
    
    attr = np.zeros((e, topA), dtype=np.float32)
    for fn in fns:
        if not os.path.exists(fn):
            continue
        try:
            with open(fn, "r", encoding="utf-8") as f:
                for line in f:
                    th = line[:-1].split("\t")
                    if len(th) >= 2 and th[0] in ent2id:
                        ent_idx = ent2id[th[0]]
                        if ent_idx < e:
                            for i in range(1, len(th)):
                                if th[i] in attr2id:
                                    attr[ent_idx][attr2id[th[i]]] = 1.0
        except Exception as e:
            print(f"Warning: Failed to load attributes from {fn}: {e}")
    return attr


def load_relation(e, KG, topR=1000):
    """加载关系特征（基础版本）"""
    try:
        rel_mat = np.zeros((e, topR), dtype=np.float32)
        rels = np.array(KG)[:, 1]
        top_rels = Counter(rels).most_common(topR)
        rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
        
        for tri in KG:
            h = tri[0]
            r = tri[1] 
            o = tri[2]
            if h < e and o < e and r in rel_index_dict:
                rel_mat[h][rel_index_dict[r]] += 1.0
                rel_mat[o][rel_index_dict[r]] += 1.0
        return np.array(rel_mat)
    except Exception as e:
        print(f"Warning: Failed to load relation features: {e}")
        return np.zeros((e, topR), dtype=np.float32)


def load_relation_enhanced(e, KG, topR=1000):
    """
    [新增] 增强的关系特征加载
    改进点：
    1. 区分作为头实体和尾实体时的关系
    2. 添加入度和出度统计
    3. 添加关系多样性特征
    4. 归一化处理
    
    特征维度: topR * 2 (头/尾) + 4 (度数统计)
    """
    print("Loading enhanced relation features...")
    
    try:
        feature_dim = topR * 2 + 4  # 头关系 + 尾关系 + 入度 + 出度 + 关系多样性(头) + 关系多样性(尾)
        rel_mat = np.zeros((e, feature_dim), dtype=np.float32)
        
        # 统计关系频率
        rels = np.array(KG)[:, 1]
        top_rels = Counter(rels).most_common(topR)
        rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
        
        # 统计度数和关系集合
        in_degree = np.zeros(e, dtype=np.float32)
        out_degree = np.zeros(e, dtype=np.float32)
        head_relations = defaultdict(set)  # 每个实体作为头时的关系集合
        tail_relations = defaultdict(set)  # 每个实体作为尾时的关系集合
        
        for tri in KG:
            h, r, o = tri[0], tri[1], tri[2]
            
            if h < e and o < e:
                out_degree[h] += 1
                in_degree[o] += 1
                
                head_relations[h].add(r)
                tail_relations[o].add(r)
                
                if r in rel_index_dict:
                    # 作为头实体时的关系（前topR维）
                    rel_mat[h][rel_index_dict[r]] += 1.0
                    # 作为尾实体时的关系（topR到2*topR维）
                    rel_mat[o][rel_index_dict[r] + topR] += 1.0
        
        # 添加度数特征（归一化）
        max_in = max(in_degree.max(), 1)
        max_out = max(out_degree.max(), 1)
        
        rel_mat[:, -4] = np.log1p(in_degree) / np.log1p(max_in)  # 归一化入度
        rel_mat[:, -3] = np.log1p(out_degree) / np.log1p(max_out)  # 归一化出度
        
        # 添加关系多样性特征
        for i in range(e):
            rel_mat[i, -2] = len(head_relations[i]) / max(topR, 1)  # 作为头时的关系多样性
            rel_mat[i, -1] = len(tail_relations[i]) / max(topR, 1)  # 作为尾时的关系多样性
        
        # 对关系频率部分进行行归一化
        row_sum = rel_mat[:, :topR*2].sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        rel_mat[:, :topR*2] = rel_mat[:, :topR*2] / row_sum
        
        print(f"Enhanced relation features: shape={rel_mat.shape}")
        print(f"  Avg in-degree: {in_degree.mean():.2f}, Avg out-degree: {out_degree.mean():.2f}")
        print(f"  Avg head relation diversity: {np.mean([len(s) for s in head_relations.values()]):.2f}")
        print(f"  Avg tail relation diversity: {np.mean([len(s) for s in tail_relations.values()]):.2f}")
        
        return rel_mat.astype(np.float32)
        
    except Exception as e:
        print(f"Warning: Failed to load enhanced relation features: {e}")
        print("Falling back to basic relation features")
        return load_relation(e, KG, topR)


def load_json_embd(path):
    """加载JSON格式的嵌入"""
    embd_dict = {}
    try:
        with open(path) as f:
            for line in f:
                example = json.loads(line.strip())
                vec = np.array([float(e) for e in example["feature"].split()])
                embd_dict[int(example["guid"])] = vec
    except Exception as e:
        print(f"Warning: Failed to load JSON embeddings from {path}: {e}")
    return embd_dict


def load_img(e_num, path):
    """加载图像特征（基础版本）"""
    if not os.path.exists(path):
        print(f"Warning: Image feature file {path} not found, creating random features")
        return np.random.normal(0, 0.1, (e_num, 2048)).astype(np.float32)
    
    try:
        img_dict = pickle.load(open(path, "rb"))
        imgs_np = np.array(list(img_dict.values()))
        
        if len(imgs_np) == 0:
            print(f"Warning: Empty image dictionary, creating random features")
            return np.random.normal(0, 0.1, (e_num, 2048)).astype(np.float32)
        
        mean = np.mean(imgs_np, axis=0)
        std = np.std(imgs_np, axis=0)
        std = np.where(std == 0, 0.1, std)
        
        img_embd = np.array([
            img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0])
            for i in range(e_num)
        ])
        print("%.2f%% entities have images" % (100 * len(img_dict) / e_num))
        return img_embd.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to load image features from {path}: {e}")
        return np.random.normal(0, 0.1, (e_num, 2048)).astype(np.float32)


def load_img_new(e_num, path, triples):
    """
    加载图像特征（增强版本）
    改进：对于缺失图像，使用邻居图像平均或全局平均
    """
    if not os.path.exists(path):
        print(f"Warning: Image feature file {path} not found, creating zero features")
        return np.zeros((e_num, 2048), dtype=np.float32)
    
    try:
        img_dict = pickle.load(open(path, "rb"))
        
        if len(img_dict) == 0:
            print(f"Warning: Empty image dictionary, creating zero features")
            return np.zeros((e_num, 2048), dtype=np.float32)
        
        # 构建邻居关系
        neighbor_list = defaultdict(list)
        for triple in triples:
            head = triple[0]
            tail = triple[2]
            if head < e_num and tail < e_num:
                if tail in img_dict:
                    neighbor_list[head].append(tail)
                if head in img_dict:
                    neighbor_list[tail].append(head)
        
        imgs_np = np.array(list(img_dict.values()))
        mean = np.mean(imgs_np, axis=0)
        feature_dim = mean.shape[0]
        
        img_embd = []
        has_real_image = []
        follow_neighbor_img_num = 0
        follow_zero_num = 0
        
        for i in range(e_num):
            if i in img_dict:
                img_embd.append(img_dict[i])
                has_real_image.append(True)
            else:
                if len(neighbor_list[i]) > 0:
                    neighbor_imgs = [img_dict[n] for n in neighbor_list[i] if n in img_dict]
                    if len(neighbor_imgs) > 0:
                        follow_neighbor_img_num += 1
                        neighbor_mean = np.mean(neighbor_imgs, axis=0)
                        img_embd.append(neighbor_mean)
                        has_real_image.append(True)
                    else:
                        follow_zero_num += 1
                        img_embd.append(mean)
                        has_real_image.append(False)
                else:
                    follow_zero_num += 1
                    img_embd.append(mean)
                    has_real_image.append(False)
        
        real_img_ratio = sum(has_real_image) / e_num * 100
        print(
            f"{100 * len(img_dict) / e_num:.2f}% entities have direct images, "
            f"{real_img_ratio:.2f}% have effective images (including neighbors), "
            f"neighbor_fill={follow_neighbor_img_num}, mean_fill={follow_zero_num}"
        )
        
        return np.array(img_embd, dtype=np.float32)
    
    except Exception as e:
        print(f"Warning: Failed to load image features from {path}: {e}")
        return np.zeros((e_num, 2048), dtype=np.float32)


def load_img_with_quality_mask(e_num, path, triples):
    """
    加载图像特征，同时返回质量掩码
    """
    if not os.path.exists(path):
        print(f"Warning: Image feature file {path} not found")
        return np.zeros((e_num, 2048), dtype=np.float32), np.zeros(e_num, dtype=np.float32)
    
    try:
        img_dict = pickle.load(open(path, "rb"))
        
        if len(img_dict) == 0:
            return np.zeros((e_num, 2048), dtype=np.float32), np.zeros(e_num, dtype=np.float32)
        
        neighbor_list = defaultdict(list)
        for triple in triples:
            head = triple[0]
            tail = triple[2]
            if head < e_num and tail < e_num:
                if tail in img_dict:
                    neighbor_list[head].append(tail)
                if head in img_dict:
                    neighbor_list[tail].append(head)
        
        imgs_np = np.array(list(img_dict.values()))
        mean = np.mean(imgs_np, axis=0)
        feature_dim = mean.shape[0]
        
        img_embd = []
        quality_mask = []
        
        for i in range(e_num):
            if i in img_dict:
                img_embd.append(img_dict[i])
                quality_mask.append(1.0)
            else:
                neighbor_imgs = [img_dict[n] for n in neighbor_list[i] if n in img_dict]
                if len(neighbor_imgs) > 0:
                    neighbor_mean = np.mean(neighbor_imgs, axis=0)
                    img_embd.append(neighbor_mean)
                    quality = min(0.8, 0.3 + 0.1 * len(neighbor_imgs))
                    quality_mask.append(quality)
                else:
                    img_embd.append(mean)
                    quality_mask.append(0.1)
        
        coverage = sum(1 for q in quality_mask if q >= 0.5) / e_num * 100
        print(f"Image coverage (quality >= 0.5): {coverage:.2f}%")
        
        return np.array(img_embd, dtype=np.float32), np.array(quality_mask, dtype=np.float32)
    
    except Exception as e:
        print(f"Warning: Failed to load image features: {e}")
        return np.zeros((e_num, 2048), dtype=np.float32), np.zeros(e_num, dtype=np.float32)


def load_entity_texts(file_dir, ent2id_dict):
    """加载实体名称用于CLIP文本编码"""
    print("Loading entity names for CLIP text encoding...")
    
    id2name = {}
    
    entity_files = [
        os.path.join(file_dir, "ent_ids_1"),
        os.path.join(file_dir, "ent_ids_2")
    ]
    
    for file_path in entity_files:
        try:
            if not os.path.exists(file_path):
                print(f"Warning: Entity file {file_path} not found")
                continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        ent_id = int(parts[0])
                        ent_name = parts[1]
                        
                        if "/" in ent_name:
                            ent_name = ent_name.split("/")[-1]
                        if "#" in ent_name:
                            ent_name = ent_name.split("#")[-1]
                        
                        ent_name = ent_name.replace("_", " ")
                        ent_name = ''.join(c for c in ent_name if c.isalnum() or c.isspace())
                        
                        if not ent_name.strip():
                            ent_name = f"entity {ent_id}"
                        
                        id2name[ent_id] = ent_name.strip()
                        
        except Exception as e:
            print(f"Warning: Failed to load entity names from {file_path}: {e}")
    
    max_id = max(ent2id_dict.values()) if ent2id_dict else 0
    entity_texts = []
    
    for i in range(max_id + 1):
        if i in id2name:
            entity_texts.append(id2name[i])
        else:
            entity_texts.append(f"entity {i}")
    
    valid_names = len([name for name in entity_texts if not name.startswith('entity ')])
    print(f"Loaded {valid_names} entity names out of {len(entity_texts)} entities")
    
    return entity_texts


def prepare_clip_text_inputs(entity_texts, max_length=77):
    """为CLIP准备文本输入"""
    processed_texts = []
    for text in entity_texts:
        if len(text) > max_length:
            text = text[:max_length]
        processed_texts.append(text)
    return processed_texts


def load_name_features(e_num, entity_files, ent2id_dict, feature_dim=300):
    """加载实体名称特征"""
    print("Loading name features...")
    
    id2name = {}
    for file_path in entity_files:
        try:
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        ent_id = int(parts[0])
                        ent_name = parts[1]
                        
                        if "/" in ent_name:
                            ent_name = ent_name.split("/")[-1]
                        if "#" in ent_name:
                            ent_name = ent_name.split("#")[-1]
                        ent_name = ent_name.replace("_", " ")
                        
                        id2name[ent_id] = ent_name.lower()
        except Exception as e:
            print(f"Warning: Failed to load names from {file_path}: {e}")
    
    features = np.zeros((e_num, feature_dim), dtype=np.float32)
    
    for i in range(e_num):
        if i in id2name:
            name = id2name[i]
            for j, char in enumerate(name[:feature_dim]):
                features[i][j % feature_dim] += ord(char) / 255.0
            norm = np.linalg.norm(features[i])
            if norm > 0:
                features[i] /= norm
    
    return features


def load_char_features(e_num, entity_files, ent2id_dict, feature_dim=100):
    """加载字符级特征"""
    print("Loading character features...")
    
    id2name = {}
    for file_path in entity_files:
        try:
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        ent_id = int(parts[0])
                        ent_name = parts[1]
                        
                        if "/" in ent_name:
                            ent_name = ent_name.split("/")[-1]
                        if "#" in ent_name:
                            ent_name = ent_name.split("#")[-1]
                        
                        id2name[ent_id] = ent_name.lower()
        except Exception as e:
            print(f"Warning: Failed to load names from {file_path}: {e}")
    
    features = np.zeros((e_num, feature_dim), dtype=np.float32)
    
    for i in range(e_num):
        if i in id2name:
            name = id2name[i]
            for n in [2, 3]:
                for j in range(len(name) - n + 1):
                    ngram = name[j:j+n]
                    idx = hash(ngram) % feature_dim
                    features[i][idx] += 1.0
            
            norm = np.linalg.norm(features[i])
            if norm > 0:
                features[i] /= norm
    
    return features


def validate_data_consistency(ent2id_dict, triples, ills):
    """验证数据一致性"""
    print("Validating data consistency...")
    
    try:
        max_ent_id = max(ent2id_dict.values()) if ent2id_dict else -1
        print(f"Max entity ID: {max_ent_id}")
        
        triple_entities = set()
        invalid_triples = 0
        for h, r, t in triples:
            if h <= max_ent_id and t <= max_ent_id:
                triple_entities.add(h)
                triple_entities.add(t)
            else:
                invalid_triples += 1
        
        if invalid_triples > 0:
            print(f"Warning: Found {invalid_triples} invalid triples")
        
        invalid_alignments = 0
        for e1, e2 in ills:
            if e1 > max_ent_id or e2 > max_ent_id:
                invalid_alignments += 1
        
        if invalid_alignments > 0:
            print(f"Warning: Found {invalid_alignments} invalid alignments")
        
        print("Data validation completed")
        return invalid_triples == 0 and invalid_alignments == 0
        
    except Exception as e:
        print(f"Warning: Data validation failed: {e}")
        return False


def create_entity_batches(entity_texts, batch_size=1000):
    """将实体文本分批处理"""
    batches = []
    for i in range(0, len(entity_texts), batch_size):
        batch = entity_texts[i:i + batch_size]
        batches.append(batch)
    return batches


def safe_save_features(features, save_path, feature_name):
    """安全保存特征"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, features)
        print(f"{feature_name} features saved to {save_path}")
        return True
    except Exception as e:
        print(f"Warning: Failed to save {feature_name} features: {e}")
        return False


def safe_load_features(load_path, feature_name):
    """安全加载特征"""
    try:
        if os.path.exists(load_path):
            features = np.load(load_path)
            print(f"{feature_name} features loaded from {load_path}")
            return features
    except Exception as e:
        print(f"Warning: Failed to load {feature_name} features from {load_path}: {e}")
    return None


def load_multimodal_data_with_cache(file_dir, ent2id_dict, ent_num, triples, 
                                   use_cache=True, cache_dir="cache",
                                   use_enhanced_rel=True):
    """
    带缓存的多模态数据加载
    [新增] use_enhanced_rel: 是否使用增强版关系特征
    """
    data_dict = {}
    
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
    
    # 图像特征
    img_cache_path = os.path.join(cache_dir, f"{os.path.basename(file_dir)}_img_features.npy")
    if use_cache and os.path.exists(img_cache_path):
        data_dict['img_features'] = safe_load_features(img_cache_path, "image")
    
    if 'img_features' not in data_dict or data_dict['img_features'] is None:
        if "V1" in file_dir:
            img_vec_path = "data/pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
        elif "V2" in file_dir:
            img_vec_path = "data/pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
        elif "FB15K" in file_dir:
            filename = os.path.split(file_dir)[-1].upper()
            img_vec_path = f"data/mmkb-datasets/{filename}/{filename}_id_img_feature_dict.pkl"
        else:
            split = file_dir.split("/")[-1]
            img_vec_path = f"data/pkls/{split}_GA_id_img_feature_dict.pkl"
        
        data_dict['img_features'] = load_img_new(ent_num, img_vec_path, triples)
        
        if use_cache:
            safe_save_features(data_dict['img_features'], img_cache_path, "image")
    
    # 属性特征
    attr_cache_path = os.path.join(cache_dir, f"{os.path.basename(file_dir)}_attr_features.npy")
    if use_cache and os.path.exists(attr_cache_path):
        data_dict['att_features'] = safe_load_features(attr_cache_path, "attribute")
    
    if 'att_features' not in data_dict or data_dict['att_features'] is None:
        a1 = os.path.join(file_dir, "training_attrs_1")
        a2 = os.path.join(file_dir, "training_attrs_2")
        data_dict['att_features'] = load_attr([a1, a2], ent_num, ent2id_dict, 1000)
        
        if use_cache:
            safe_save_features(data_dict['att_features'], attr_cache_path, "attribute")
    
    # [改进] 关系特征 - 支持增强版
    rel_suffix = "_enhanced" if use_enhanced_rel else ""
    rel_cache_path = os.path.join(cache_dir, f"{os.path.basename(file_dir)}_rel_features{rel_suffix}.npy")
    if use_cache and os.path.exists(rel_cache_path):
        data_dict['rel_features'] = safe_load_features(rel_cache_path, "relation")
    
    if 'rel_features' not in data_dict or data_dict['rel_features'] is None:
        if use_enhanced_rel:
            data_dict['rel_features'] = load_relation_enhanced(ent_num, triples, 1000)
        else:
            data_dict['rel_features'] = load_relation(ent_num, triples, 1000)
        
        if use_cache:
            safe_save_features(data_dict['rel_features'], rel_cache_path, "relation")
    
    # 实体文本
    text_cache_path = os.path.join(cache_dir, f"{os.path.basename(file_dir)}_entity_texts.json")
    if use_cache and os.path.exists(text_cache_path):
        try:
            with open(text_cache_path, 'r', encoding='utf-8') as f:
                data_dict['entity_texts'] = json.load(f)
            print("Entity texts loaded from cache")
        except Exception as e:
            print(f"Warning: Failed to load cached entity texts: {e}")
            data_dict['entity_texts'] = load_entity_texts(file_dir, ent2id_dict)
    else:
        data_dict['entity_texts'] = load_entity_texts(file_dir, ent2id_dict)
        if use_cache:
            try:
                with open(text_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data_dict['entity_texts'], f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to cache entity texts: {e}")
    
    return data_dict


def create_fallback_features(ent_num, feature_name, feature_dim):
    """创建备用特征"""
    print(f"Creating fallback {feature_name} features with dimension {feature_dim}")
    return np.random.normal(0, 0.1, (ent_num, feature_dim)).astype(np.float32)


def check_feature_integrity(features, feature_name, expected_shape):
    """检查特征完整性"""
    if features is None:
        print(f"Warning: {feature_name} features are None")
        return False
    
    if features.shape != expected_shape:
        print(f"Warning: {feature_name} features shape {features.shape} doesn't match expected {expected_shape}")
        return False
    
    if np.isnan(features).any():
        print(f"Warning: {feature_name} features contain NaN values")
        return False
    
    if np.isinf(features).any():
        print(f"Warning: {feature_name} features contain infinite values")
        return False
    
    print(f"{feature_name} features integrity check passed")
    return True


def analyze_image_coverage(e_num, path, left_ents, right_ents):
    """分析图像覆盖情况"""
    if not os.path.exists(path):
        print(f"Image file not found: {path}")
        return
    
    try:
        img_dict = pickle.load(open(path, "rb"))
        
        left_coverage = sum(1 for e in left_ents if e in img_dict) / len(left_ents) * 100
        right_coverage = sum(1 for e in right_ents if e in img_dict) / len(right_ents) * 100
        
        print(f"Image coverage analysis:")
        print(f"  Left KG: {left_coverage:.2f}% ({sum(1 for e in left_ents if e in img_dict)}/{len(left_ents)})")
        print(f"  Right KG: {right_coverage:.2f}% ({sum(1 for e in right_ents if e in img_dict)}/{len(right_ents)})")
        print(f"  Total: {len(img_dict)}/{e_num} ({len(img_dict)/e_num*100:.2f}%)")
        
        return {
            'left_coverage': left_coverage,
            'right_coverage': right_coverage,
            'total': len(img_dict) / e_num * 100
        }
        
    except Exception as e:
        print(f"Failed to analyze image coverage: {e}")
        return None
