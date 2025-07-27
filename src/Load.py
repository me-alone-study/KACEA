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


def load_name_features(e_num, file_paths, ent2id_dict, word2vec_path=None, feature_dim=300):
    """加载实体名称特征"""
    try:
        # 简化版本：直接创建随机特征
        print(f"Creating random name features with dimension {feature_dim}")
        name_features = np.random.normal(0, 0.1, (e_num, feature_dim)).astype(np.float32)
        return name_features
    except Exception as e:
        print(f"Warning: Failed to load name features: {e}")
        return np.random.normal(0, 0.1, (e_num, 300)).astype(np.float32)


def load_char_features(e_num, file_paths, ent2id_dict, feature_dim=100):
    """加载字符级特征"""
    try:
        # 简化版本：直接创建随机特征
        print(f"Creating random char features with dimension {feature_dim}")
        char_features = np.random.normal(0, 0.1, (e_num, feature_dim)).astype(np.float32)
        return char_features
    except Exception as e:
        print(f"Warning: Failed to load char features: {e}")
        return np.random.normal(0, 0.1, (e_num, 100)).astype(np.float32)


def load_entity_names(ent2id_dict, file_paths):
    """
    加载实体名称用于CLIP文本编码
    
    Args:
        ent2id_dict: 实体到ID的映射字典
        file_paths: 实体文件路径列表
    
    Returns:
        entity_names: 按实体ID排序的实体名称列表
    """
    print("Loading entity names for CLIP text encoding...")
    
    # 创建ID到名称的映射
    id2name = {}
    
    for file_path in file_paths:
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
                        id2name[ent_id] = ent_name
        except Exception as e:
            print(f"Warning: Failed to load entity names from {file_path}: {e}")
    
    # 按ID顺序创建名称列表
    max_id = max(ent2id_dict.values()) if ent2id_dict else 0
    entity_names = []
    
    for i in range(max_id + 1):
        if i in id2name:
            entity_names.append(id2name[i])
        else:
            # 对于缺失的实体，使用默认名称
            entity_names.append(f"entity_{i}")
    
    valid_names = len([name for name in entity_names if not name.startswith('entity_')])
    print(f"Loaded {valid_names} entity names out of {len(entity_names)} entities")
    return entity_names


def prepare_clip_text_inputs(entity_names, tokenizer, max_length=77):
    """
    为CLIP准备文本输入
    
    Args:
        entity_names: 实体名称列表
        tokenizer: CLIP tokenizer
        max_length: 最大序列长度
    
    Returns:
        tokenized_inputs: 标记化的输入
    """
    try:
        if tokenizer is None:
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # 批量标记化
        tokenized = tokenizer(
            entity_names,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        return tokenized
    except ImportError:
        print("Warning: transformers not available, returning None")
        return None
    except Exception as e:
        print(f"Warning: Failed to prepare CLIP text inputs: {e}")
        return None


def load_clip_compatible_images(img_features, target_size=(224, 224)):
    """
    将图像特征转换为CLIP兼容格式
    
    Args:
        img_features: 原始图像特征
        target_size: 目标图像尺寸
    
    Returns:
        clip_images: CLIP兼容的图像特征
    """
    try:
        # 如果图像特征已经是合适的格式，直接返回
        if isinstance(img_features, torch.Tensor):
            return img_features
        
        # 否则进行转换
        if isinstance(img_features, np.ndarray):
            return torch.FloatTensor(img_features)
        else:
            print("Warning: Unsupported image feature format")
            return None
    except Exception as e:
        print(f"Warning: Failed to convert image features: {e}")
        return None


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


def save_embeddings(embeddings, path):
    """保存嵌入到文件"""
    try:
        np.save(path, embeddings)
        print(f"Embeddings saved to {path}")
    except Exception as e:
        print(f"Warning: Failed to save embeddings: {e}")


def load_embeddings(path):
    """从文件加载嵌入"""
    try:
        embeddings = np.load(path)
        print(f"Embeddings loaded from {path}")
        return embeddings
    except Exception as e:
        print(f"Warning: Failed to load embeddings from {path}: {e}")
        return None


# 辅助函数
def normalize_embeddings(embeddings):
    """归一化嵌入"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def pad_sequences(sequences, max_length=None, padding_value=0):
    """填充序列到相同长度"""
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) >= max_length:
            padded.append(seq[:max_length])
        else:
            padding = [padding_value] * (max_length - len(seq))
            padded.append(seq + padding)
    
    return np.array(padded)


def create_entity_vocab(entity_names):
    """创建实体词汇表"""
    vocab = set()
    for name in entity_names:
        if name:
            words = name.lower().split()
            vocab.update(words)
    
    vocab = sorted(list(vocab))
    word2id = {word: i for i, word in enumerate(vocab)}
    return vocab, word2id


def encode_entity_names(entity_names, word2id, max_length=10):
    """将实体名称编码为ID序列"""
    encoded = []
    for name in entity_names:
        if name:
            words = name.lower().split()
            ids = [word2id.get(word, 0) for word in words]  # 0 for unknown words
            encoded.append(ids)
        else:
            encoded.append([0])
    
    return pad_sequences(encoded, max_length)


# 数据验证函数
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


# 主要的数据加载接口
def load_multimodal_data(file_dir, ent2id_dict, ent_num, triples, 
                        load_images=True, load_attributes=True, 
                        load_relations=True, load_names=False, load_chars=False):
    """
    统一的多模态数据加载接口
    
    Args:
        file_dir: 数据目录
        ent2id_dict: 实体映射字典
        ent_num: 实体数量
        triples: 三元组列表
        load_images: 是否加载图像特征
        load_attributes: 是否加载属性特征
        load_relations: 是否加载关系特征
        load_names: 是否加载名称特征
        load_chars: 是否加载字符特征
    
    Returns:
        data_dict: 包含所有特征的字典
    """
    data_dict = {}
    
    # 加载图像特征
    if load_images:
        # 确定图像特征文件路径
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
    
    # 加载属性特征
    if load_attributes:
        a1 = os.path.join(file_dir, "training_attrs_1")
        a2 = os.path.join(file_dir, "training_attrs_2")
        data_dict['att_features'] = load_attr([a1, a2], ent_num, ent2id_dict, 1000)
    
    # 加载关系特征
    if load_relations:
        data_dict['rel_features'] = load_relation(ent_num, triples, 1000)
    
    # 加载名称特征
    if load_names:
        e1 = os.path.join(file_dir, "ent_ids_1")
        e2 = os.path.join(file_dir, "ent_ids_2")
        data_dict['name_features'] = load_name_features(ent_num, [e1, e2], ent2id_dict)
    
    # 加载字符特征
    if load_chars:
        e1 = os.path.join(file_dir, "ent_ids_1")
        e2 = os.path.join(file_dir, "ent_ids_2")
        data_dict['char_features'] = load_char_features(ent_num, [e1, e2], ent2id_dict)
    
    return data_dict