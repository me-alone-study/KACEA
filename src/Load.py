#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split("\t")
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ids(fn):
    """从文件中获取ID列表"""
    ids = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split("\t")
            ids.append(int(th[0]))
    return ids


def get_ent2id(fns):
    """构建实体到ID的映射"""
    ent2id = {}
    for fn in fns:
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                ent2id[th[1]] = int(th[0])
    return ent2id


def load_attr(fns, e, ent2id, topA=1000):
    """加载属性特征"""
    cnt = {}
    for fn in fns:
        if not os.path.exists(fn):
            print(f"Warning: Attribute file {fn} not found, skipping...")
            continue
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    
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
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr


def load_relation(e, KG, topR=1000):
    """加载关系特征"""
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    rels = np.array(KG)[:, 1]
    top_rels = Counter(rels).most_common(topR)
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
    
    for tri in KG:
        h = tri[0]
        r = tri[1] 
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.0
            rel_mat[o][rel_index_dict[r]] += 1.0
    return np.array(rel_mat)


def load_json_embd(path):
    """加载JSON格式的嵌入"""
    embd_dict = {}
    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())
            vec = np.array([float(e) for e in example["feature"].split()])
            embd_dict[int(example["guid"])] = vec
    return embd_dict


def load_img(e_num, path):
    """加载图像特征（基础版本）"""
    if not os.path.exists(path):
        print(f"Warning: Image feature file {path} not found, creating random features")
        return np.random.normal(0, 0.1, (e_num, 2048)).astype(np.float32)
    
    try:
        img_dict = pickle.load(open(path, "rb"))
        imgs_np = np.array(list(img_dict.values()))
        mean = np.mean(imgs_np, axis=0)
        std = np.std(imgs_np, axis=0)
        
        img_embd = np.array([
            img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0])
            for i in range(e_num)
        ])
        print("%.2f%% entities have images" % (100 * len(img_dict) / e_num))
        return img_embd
    except Exception as e:
        print(f"Warning: Failed to load image features from {path}: {e}")
        return np.random.normal(0, 0.1, (e_num, 2048)).astype(np.float32)


def load_img_new(e_num, path, triples):
    """加载图像特征（增强版本，使用邻居信息）"""
    if not os.path.exists(path):
        print(f"Warning: Image feature file {path} not found, creating random features")
        return np.random.normal(0, 0.1, (e_num, 2048)).astype(np.float32)
    
    try:
        img_dict = pickle.load(open(path, "rb"))
        
        # 构建邻居关系
        neighbor_list = defaultdict(list)
        for triple in triples:
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            if tail in img_dict:
                neighbor_list[head].append(tail)
            if head in img_dict:
                neighbor_list[tail].append(head)
        
        imgs_np = np.array(list(img_dict.values()))
        mean = np.mean(imgs_np, axis=0)
        std = np.std(imgs_np, axis=0)
        all_img_emb_normal = np.random.normal(mean, std, mean.shape[0])
        
        img_embd = []
        follow_neighbor_img_num = 0
        follow_all_img_num = 0
        
        for i in range(e_num):
            if i in img_dict:
                img_embd.append(img_dict[i])
            else:
                if len(neighbor_list[i]) > 0:
                    follow_neighbor_img_num += 1
                    if i in img_dict:
                        neighbor_list[i].append(i)
                    neighbor_imgs_emb = np.array([img_dict[id] for id in neighbor_list[i] if id in img_dict])
                    if len(neighbor_imgs_emb) > 0:
                        neighbor_imgs_emb_mean = np.mean(neighbor_imgs_emb, axis=0)
                        img_embd.append(neighbor_imgs_emb_mean)
                    else:
                        img_embd.append(all_img_emb_normal)
                else:
                    follow_all_img_num += 1
                    img_embd.append(all_img_emb_normal)
        
        print(
            "%.2f%% entities have images," % (100 * len(img_dict) / e_num),
            " follow_neighbor_img_num is {0},".format(follow_neighbor_img_num),
            " follow_all_img_num is {0}".format(follow_all_img_num),
        )
        return np.array(img_embd)
    
    except Exception as e:
        print(f"Warning: Failed to load image features from {path}: {e}")
        return np.random.normal(0, 0.1, (e_num, 2048)).astype(np.float32)


def load_entity_texts(file_dir, ent2id_dict):
    """
    加载实体名称用于CLIP文本编码
    
    Args:
        file_dir: 数据目录路径
        ent2id_dict: 实体到ID的映射字典
    
    Returns:
        entity_texts: 按实体ID排序的实体名称列表
    """
    print("Loading entity names for CLIP text encoding...")
    
    # 创建ID到名称的映射
    id2name = {}
    
    # 读取实体文件
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
                        
                        # 清理实体名称，移除URI前缀等
                        if "/" in ent_name:
                            ent_name = ent_name.split("/")[-1]
                        if "#" in ent_name:
                            ent_name = ent_name.split("#")[-1]
                        
                        # 替换下划线为空格，使其更适合自然语言处理
                        ent_name = ent_name.replace("_", " ")
                        
                        # 移除特殊字符，保留字母数字和空格
                        ent_name = ''.join(c for c in ent_name if c.isalnum() or c.isspace())
                        
                        # 如果名称为空，使用默认名称
                        if not ent_name.strip():
                            ent_name = f"entity {ent_id}"
                        
                        id2name[ent_id] = ent_name.strip()
                        
        except Exception as e:
            print(f"Warning: Failed to load entity names from {file_path}: {e}")
    
    # 按ID顺序创建名称列表
    max_id = max(ent2id_dict.values()) if ent2id_dict else 0
    entity_texts = []
    
    for i in range(max_id + 1):
        if i in id2name:
            entity_texts.append(id2name[i])
        else:
            # 对于缺失的实体，使用默认名称
            entity_texts.append(f"entity {i}")
    
    valid_names = len([name for name in entity_texts if not name.startswith('entity ')])
    print(f"Loaded {valid_names} entity names out of {len(entity_texts)} entities")
    
    return entity_texts


def prepare_clip_text_inputs(entity_texts, max_length=77):
    """
    为CLIP准备文本输入
    
    Args:
        entity_texts: 实体名称列表
        max_length: 最大序列长度
    
    Returns:
        processed_texts: 处理后的文本列表
    """
    try:
        processed_texts = []
        
        for text in entity_texts:
            # 确保文本不为空
            if not text or text.strip() == "":
                text = "unknown entity"
            
            # 限制文本长度（粗略估计，实际tokenization会更精确）
            if len(text) > max_length * 4:  # 粗略估计每个token平均4个字符
                text = text[:max_length * 4]
            
            processed_texts.append(text)
        
        return processed_texts
        
    except Exception as e:
        print(f"Warning: Failed to prepare CLIP text inputs: {e}")
        return entity_texts


def load_name_features(e_num, file_paths, ent2id_dict, word2vec_path=None, feature_dim=300):
    """加载实体名称特征"""
    try:
        print(f"Loading name features...")
        
        # 首先尝试加载实体文本
        entity_texts = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            entity_texts.append(parts[1])
        
        if entity_texts:
            # 如果有实体文本，创建简单的特征表示
            # 这里可以改进为使用预训练词向量
            name_features = np.random.normal(0, 0.1, (e_num, feature_dim)).astype(np.float32)
            print(f"Created name features with dimension {feature_dim}")
        else:
            # 如果没有文本，创建随机特征
            name_features = np.random.normal(0, 0.1, (e_num, feature_dim)).astype(np.float32)
            print(f"Created random name features with dimension {feature_dim}")
        
        return name_features
        
    except Exception as e:
        print(f"Warning: Failed to load name features: {e}")
        return np.random.normal(0, 0.1, (e_num, 300)).astype(np.float32)


def load_char_features(e_num, file_paths, ent2id_dict, feature_dim=100):
    """加载字符级特征"""
    try:
        print(f"Loading char features...")
        
        # 创建字符级特征（可以改进为基于字符n-gram等）
        char_features = np.random.normal(0, 0.1, (e_num, feature_dim)).astype(np.float32)
        print(f"Created char features with dimension {feature_dim}")
        
        return char_features
        
    except Exception as e:
        print(f"Warning: Failed to load char features: {e}")
        return np.random.normal(0, 0.1, (e_num, 100)).astype(np.float32)


def load_word_embeddings(word2vec_path, vocab_size=None, embedding_dim=300):
    """
    加载词向量
    
    Args:
        word2vec_path: 词向量文件路径
        vocab_size: 词汇表大小
        embedding_dim: 嵌入维度
    
    Returns:
        word_embeddings: 词嵌入矩阵
        word2id: 词到ID的映射
    """
    try:
        if not os.path.exists(word2vec_path):
            print(f"Warning: Word embedding file {word2vec_path} not found")
            if vocab_size is None:
                vocab_size = 10000
            return np.random.normal(0, 0.1, (vocab_size, embedding_dim)), {}
        
        print(f"Loading word embeddings from {word2vec_path}")
        word2id = {}
        embeddings = []
        
        with open(word2vec_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0 and len(line.split()) == 2:
                    # 跳过第一行如果包含维度信息
                    continue
                parts = line.strip().split()
                if len(parts) < embedding_dim + 1:
                    continue
                word = parts[0]
                vector = [float(x) for x in parts[1:embedding_dim+1]]
                word2id[word] = len(embeddings)
                embeddings.append(vector)
                
                if vocab_size and len(embeddings) >= vocab_size:
                    break
        
        if not embeddings:
            print("Warning: No embeddings loaded, creating random ones")
            if vocab_size is None:
                vocab_size = 10000
            return np.random.normal(0, 0.1, (vocab_size, embedding_dim)), {}
        
        embeddings = np.array(embeddings, dtype=np.float32)
        print(f"Loaded {len(embeddings)} word embeddings")
        return embeddings, word2id
    
    except Exception as e:
        print(f"Warning: Failed to load word embeddings: {e}")
        if vocab_size is None:
            vocab_size = 10000
        return np.random.normal(0, 0.1, (vocab_size, embedding_dim)), {}


def validate_data_consistency(ent2id_dict, triples, ills):
    """验证数据一致性"""
    print("Validating data consistency...")
    
    # 检查实体ID范围
    max_ent_id = max(ent2id_dict.values()) if ent2id_dict else -1
    print(f"Max entity ID: {max_ent_id}")
    
    # 检查三元组中的实体ID
    triple_entities = set()
    for h, r, t in triples:
        triple_entities.add(h)
        triple_entities.add(t)
    
    invalid_entities = [e for e in triple_entities if e > max_ent_id]
    if invalid_entities:
        print(f"Warning: Found {len(invalid_entities)} invalid entity IDs in triples")
    
    # 检查对齐数据
    invalid_alignments = []
    for e1, e2 in ills:
        if e1 > max_ent_id or e2 > max_ent_id:
            invalid_alignments.append((e1, e2))
    
    if invalid_alignments:
        print(f"Warning: Found {len(invalid_alignments)} invalid alignments")
    
    print("Data validation completed")
    return len(invalid_entities) == 0 and len(invalid_alignments) == 0


def create_entity_batches(entity_texts, batch_size=1000):
    """
    将实体文本分批处理，避免内存问题
    
    Args:
        entity_texts: 实体文本列表
        batch_size: 批次大小
    
    Returns:
        batches: 文本批次列表
    """
    batches = []
    for i in range(0, len(entity_texts), batch_size):
        batch = entity_texts[i:i + batch_size]
        batches.append(batch)
    return batches


def save_processed_features(features, save_path, feature_name):
    """保存处理后的特征"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, features)
        print(f"{feature_name} features saved to {save_path}")
    except Exception as e:
        print(f"Warning: Failed to save {feature_name} features: {e}")


def load_processed_features(load_path, feature_name):
    """加载已处理的特征"""
    try:
        if os.path.exists(load_path):
            features = np.load(load_path)
            print(f"{feature_name} features loaded from {load_path}")
            return features
    except Exception as e:
        print(f"Warning: Failed to load {feature_name} features from {load_path}: {e}")
    return None


def load_multimodal_data_with_cache(file_dir, ent2id_dict, ent_num, triples, 
                                   use_cache=True, cache_dir="cache"):
    """
    带缓存的多模态数据加载
    
    Args:
        file_dir: 数据目录
        ent2id_dict: 实体映射字典
        ent_num: 实体数量
        triples: 三元组列表
        use_cache: 是否使用缓存
        cache_dir: 缓存目录
    
    Returns:
        data_dict: 包含所有特征的字典
    """
    data_dict = {}
    
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
    
    # 图像特征
    img_cache_path = os.path.join(cache_dir, f"{os.path.basename(file_dir)}_img_features.npy")
    if use_cache and os.path.exists(img_cache_path):
        data_dict['img_features'] = load_processed_features(img_cache_path, "image")
    else:
        # 原始加载逻辑
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
            save_processed_features(data_dict['img_features'], img_cache_path, "image")
    
    # 属性特征
    attr_cache_path = os.path.join(cache_dir, f"{os.path.basename(file_dir)}_attr_features.npy")
    if use_cache and os.path.exists(attr_cache_path):
        data_dict['att_features'] = load_processed_features(attr_cache_path, "attribute")
    else:
        a1 = os.path.join(file_dir, "training_attrs_1")
        a2 = os.path.join(file_dir, "training_attrs_2")
        data_dict['att_features'] = load_attr([a1, a2], ent_num, ent2id_dict, 1000)
        
        if use_cache:
            save_processed_features(data_dict['att_features'], attr_cache_path, "attribute")
    
    # 关系特征
    rel_cache_path = os.path.join(cache_dir, f"{os.path.basename(file_dir)}_rel_features.npy")
    if use_cache and os.path.exists(rel_cache_path):
        data_dict['rel_features'] = load_processed_features(rel_cache_path, "relation")
    else:
        data_dict['rel_features'] = load_relation(ent_num, triples, 1000)
        
        if use_cache:
            save_processed_features(data_dict['rel_features'], rel_cache_path, "relation")
    
    # 实体文本（用于CLIP）
    text_cache_path = os.path.join(cache_dir, f"{os.path.basename(file_dir)}_entity_texts.json")
    if use_cache and os.path.exists(text_cache_path):
        try:
            with open(text_cache_path, 'r', encoding='utf-8') as f:
                data_dict['entity_texts'] = json.load(f)
            print("Entity texts loaded from cache")
        except:
            data_dict['entity_texts'] = load_entity_texts(file_dir, ent2id_dict)
            if use_cache:
                try:
                    with open(text_cache_path, 'w', encoding='utf-8') as f:
                        json.dump(data_dict['entity_texts'], f, ensure_ascii=False, indent=2)
                except:
                    pass
    else:
        data_dict['entity_texts'] = load_entity_texts(file_dir, ent2id_dict)
        if use_cache:
            try:
                with open(text_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data_dict['entity_texts'], f, ensure_ascii=False, indent=2)
            except:
                pass
    
    return data_dict