import numpy as np
from collections import Counter
import json
import pickle
from tqdm import tqdm
import os
from PIL import Image
import torch
import clip

def loadfile(fn, num=1):
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
    ids = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split("\t")
            ids.append(int(th[0]))
    return ids


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                ent2id[th[1]] = int(th[0])
    return ent2id


def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}
    for fn in fns:
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
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    for i in range(min(topA, len(fre))):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)
    for fn in fns:
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr


def load_relation(e, KG, topR=1000):
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
    embd_dict = {}
    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())
            vec = np.array([float(e) for e in example["feature"].split()])
            embd_dict[int(example["guid"])] = vec
    return embd_dict


def load_img(e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    img_embd = np.array(
        [
            img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0])
            for i in range(e_num)
        ]
    )
    print("%.2f%% entities have images" % (100 * len(img_dict) / e_num))
    return img_embd


def load_img_new(e_num, path, triples):
    from collections import defaultdict

    img_dict = pickle.load(open(path, "rb"))
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
    follow_neirbor_img_num = 0
    follow_all_img_num = 0
    for i in range(e_num):
        if i in img_dict:
            img_embd.append(img_dict[i])
        else:
            if len(neighbor_list[i]) > 0:
                follow_neirbor_img_num += 1
                if i in img_dict:
                    neighbor_list[i].append(i)
                neighbor_imgs_emb = np.array([img_dict[id] for id in neighbor_list[i]])
                neighbor_imgs_emb_mean = np.mean(neighbor_imgs_emb, axis=0)
                img_embd.append(neighbor_imgs_emb_mean)
            else:
                follow_all_img_num += 1
                img_embd.append(all_img_emb_normal)
    print(
        "%.2f%% entities have images," % (100 * len(img_dict) / e_num),
        " follow_neirbor_img_num is {0},".format(follow_neirbor_img_num),
        " follow_all_img_num is {0}".format(follow_all_img_num),
    )
    return np.array(img_embd)


# === 新增CLIP相关数据加载函数 ===

def load_clip_data(file_dir, ent2id_dict):
    """
    加载用于CLIP的原始图像和文本数据
    
    Args:
        file_dir: 数据集目录
        ent2id_dict: 实体到ID的映射字典
    
    Returns:
        text_descriptions: {ent_id: description}
        image_paths: {ent_id: image_path}
    """
    text_descriptions = {}
    image_paths = {}
    
    # 加载实体描述文本
    desc_files = [
        os.path.join(file_dir, "entity_descriptions.txt"),
        os.path.join(file_dir, "ent_descriptions_1"),
        os.path.join(file_dir, "ent_descriptions_2"),
    ]
    
    for desc_file in desc_files:
        if os.path.exists(desc_file):
            print(f"Loading entity descriptions from {desc_file}")
            with open(desc_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ent_name = parts[0]
                        description = parts[1]
                        if ent_name in ent2id_dict:
                            ent_id = ent2id_dict[ent_name]
                            text_descriptions[ent_id] = description
    
    # 如果没有描述文件，使用实体ID作为默认描述
    if not text_descriptions:
        print("No entity descriptions found, using entity IDs as descriptions")
        for ent_name, ent_id in ent2id_dict.items():
            # 简单的实体名称清理
            clean_name = ent_name.replace('_', ' ').replace('/', ' ')
            text_descriptions[ent_id] = f"entity {clean_name}"
    
    # 加载图像路径
    img_path_files = [
        os.path.join(file_dir, "image_paths.txt"),
        os.path.join(file_dir, "images.txt"),
    ]
    
    for img_file in img_path_files:
        if os.path.exists(img_file):
            print(f"Loading image paths from {img_file}")
            with open(img_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ent_name = parts[0]
                        img_path = parts[1]
                        if ent_name in ent2id_dict:
                            ent_id = ent2id_dict[ent_name]
                            # 确保图像路径是绝对路径
                            if not os.path.isabs(img_path):
                                img_path = os.path.join(file_dir, img_path)
                            image_paths[ent_id] = img_path
    
    print(f"Loaded {len(text_descriptions)} text descriptions")
    print(f"Loaded {len(image_paths)} image paths")
    
    return text_descriptions, image_paths


def preprocess_clip_images(image_paths, clip_preprocess, device, max_entities=None):
    """
    预处理图像用于CLIP
    
    Args:
        image_paths: {ent_id: image_path}
        clip_preprocess: CLIP预处理函数
        device: 设备
        max_entities: 最大处理实体数（用于调试）
    
    Returns:
        processed_images: {ent_id: preprocessed_tensor}
    """
    processed_images = {}
    error_count = 0
    
    # 限制处理的实体数量（用于调试）
    if max_entities:
        image_paths = dict(list(image_paths.items())[:max_entities])
    
    print(f"Preprocessing {len(image_paths)} images for CLIP...")
    
    for ent_id, img_path in tqdm(image_paths.items(), desc="Processing images"):
        try:
            if os.path.exists(img_path):
                # 加载和预处理图像
                image = Image.open(img_path).convert('RGB')
                processed_img = clip_preprocess(image).unsqueeze(0).to(device)
                processed_images[ent_id] = processed_img
            else:
                error_count += 1
                if error_count <= 10:  # 只打印前10个错误
                    print(f"Image not found: {img_path}")
        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"Error processing image for entity {ent_id}: {e}")
    
    if error_count > 10:
        print(f"... and {error_count - 10} more image processing errors")
    
    print(f"Successfully processed {len(processed_images)} images")
    return processed_images


def create_clip_batch_data(entity_indices, text_descriptions, processed_images, device):
    """
    为给定的实体索引创建CLIP批次数据
    
    Args:
        entity_indices: 实体索引列表
        text_descriptions: {ent_id: description}
        processed_images: {ent_id: preprocessed_tensor}
        device: 设备
    
    Returns:
        batch_texts: 文本描述列表
        batch_images: 图像张量或None
    """
    batch_texts = []
    batch_images = []
    
    for idx in entity_indices:
        idx_item = idx.item() if torch.is_tensor(idx) else idx
        
        # 文本描述
        text_desc = text_descriptions.get(idx_item, f"entity {idx_item}")
        batch_texts.append(text_desc)
        
        # 图像
        if idx_item in processed_images:
            batch_images.append(processed_images[idx_item])
        else:
            # 使用零张量作为占位符
            batch_images.append(torch.zeros(1, 3, 224, 224).to(device))
    
    # 转换为张量
    if batch_images:
        try:
            batch_images_tensor = torch.cat(batch_images, dim=0)
        except Exception as e:
            print(f"Error creating image batch: {e}")
            batch_images_tensor = None
    else:
        batch_images_tensor = None
    
    return batch_texts, batch_images_tensor


def load_entity_names(file_dir, ent2id_dict):
    """
    加载实体名称用于生成文本描述
    
    Args:
        file_dir: 数据集目录
        ent2id_dict: 实体到ID的映射
    
    Returns:
        entity_names: {ent_id: name}
    """
    entity_names = {}
    
    # 尝试从ent2id_dict反向获取名称
    for ent_name, ent_id in ent2id_dict.items():
        # 清理实体名称用作描述
        clean_name = ent_name.replace('_', ' ').replace('/', ' ').replace('<', '').replace('>', '')
        entity_names[ent_id] = clean_name
    
    print(f"Generated names for {len(entity_names)} entities")
    return entity_names


def save_clip_cache(clip_data, cache_path):
    """
    保存CLIP数据到缓存文件
    """
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(clip_data, f)
        print(f"CLIP data cached to {cache_path}")
    except Exception as e:
        print(f"Error saving CLIP cache: {e}")


def load_clip_cache(cache_path):
    """
    从缓存文件加载CLIP数据
    """
    try:
        with open(cache_path, 'rb') as f:
            clip_data = pickle.load(f)
        print(f"CLIP data loaded from cache: {cache_path}")
        return clip_data
    except Exception as e:
        print(f"Error loading CLIP cache: {e}")
        return None
