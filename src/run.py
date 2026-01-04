#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import gc
import json
from datetime import datetime
from pprint import pprint
from transformers import (
    get_cosine_schedule_with_warmup,
)
import torch.optim as optim

try:
    from utils import *
    from models import *
    from Load import *
    from loss import *
except:
    from src.utils import *
    from src.models import *
    from src.Load import *
    from src.loss import *

# CLIP相关导入
try:
    from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available, CLIP features disabled")
    CLIPModel = CLIPTokenizer = CLIPProcessor = None
    CLIP_AVAILABLE = False


def load_img_features(ent_num, file_dir, triples, use_mean_img=False):
    if "V1" in file_dir:
        split = "norm"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
    elif "V2" in file_dir:
        split = "dense"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
    elif "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        img_vec_path = (
            "data/mmkb-datasets/"
            + filename
            + "/"
            + filename
            + "_id_img_feature_dict.pkl"
        )
    else:
        split = file_dir.split("/")[-1]
        img_vec_path = "data/pkls/" + split + "_GA_id_img_feature_dict.pkl"
    if use_mean_img:
        img_features = load_img(ent_num, img_vec_path)
    else:
        img_features = load_img_new(ent_num, img_vec_path, triples)
    return img_features


def load_img_features_dropout(
    ent_num, file_dir, triples, use_mean_img=False, img_dp_ratio=1.0
):
    if "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        if abs(1.0 - img_dp_ratio) > 1e-3:
            img_vec_path = (
                "data/mmkb-datasets/"
                + "mmkb_dropout"
                + "/"
                + filename
                + "_id_img_feature_dict_with_dropout{0}.pkl".format(img_dp_ratio)
            )
            print("dropout img_vec_path: ", img_vec_path)
        else:
            img_vec_path = (
                "data/mmkb-datasets/"
                + filename
                + "/"
                + filename
                + "_id_img_feature_dict.pkl"
            )
    else:
        split = file_dir.split("/")[-1]
        if abs(1.0 - img_dp_ratio) > 1e-3:
            img_vec_path = (
                "data/pkls/dbp_dropout/"
                + split
                + "_GA_id_img_feature_dict_with_dropout{0}.pkl".format(img_dp_ratio)
            )
        else:
            img_vec_path = "data/pkls/" + split + "_GA_id_img_feature_dict.pkl"
    if use_mean_img:
        img_features = load_img(ent_num, img_vec_path)
    else:
        img_features = load_img_new(ent_num, img_vec_path, triples)
    return img_features


class KACEA:

    def __init__(self):
        # 数据相关属性
        self.ent2id_dict = None
        self.ills = None
        self.triples = None
        self.r_hs = None
        self.r_ts = None
        self.ids = None
        self.left_ents = None
        self.right_ents = None

        # 特征相关属性
        self.img_features = None
        self.rel_features = None
        self.att_features = None
        self.char_features = None
        self.name_features = None
        self.entity_texts = None  # 新增：实体文本用于CLIP
        
        # 训练相关属性
        self.ent_vec = None
        self.left_non_train = None
        self.right_non_train = None
        self.ENT_NUM = None
        self.REL_NUM = None
        self.adj = None
        self.train_ill = None
        self.test_ill_ = None
        self.test_ill = None
        self.test_left = None
        self.test_right = None
        
        # 模型相关属性
        self.multimodal_encoder = None
        self.params = None
        self.optimizer = None
        self.scheduler = None
        
        # 损失函数
        self.criterion_ms = None
        self.criterion_nce = None
        self.comprehensive_loss = None
        
        # 实验管理
        self.parser = argparse.ArgumentParser()
        self.args = self.parse_options(self.parser)
        self.set_seed(self.args.seed, self.args.cuda)
        self.device = torch.device(
            "cuda" if self.args.cuda and torch.cuda.is_available() else "cpu"
        )
        
        # 初始化日志系统
        self.log_dir, self.log_file = self.setup_logging()
        self.exp_results = []
        
        # 最佳结果记录
        self.best_hit_1 = 0.0
        self.best_epoch = 0
        self.best_data_list = []
        self.best_to_write = []
        
        # 初始化
        self.init_data()
        self.init_model()
        self.print_summary()

    def setup_logging(self):
        """设置日志目录和文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset = os.path.basename(self.args.file_dir)
        
        exp_name = f"{dataset}_{timestamp}"
        if self.args.use_clip:
            exp_name += "_clip"
        if self.args.use_entity_names:
            exp_name += "_names"
        if getattr(self.args, 'clip_freeze', True):
            exp_name += "_frozen"
        else:
            exp_name += "_finetune"
            
        # 创建logs文件夹（与src同级）
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        log_dir = os.path.join(logs_dir, exp_name)
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, "training.log")
        
        # 保存配置
        config = {k: v for k, v in vars(self.args).items() 
                 if isinstance(v, (str, int, float, bool))}
        with open(os.path.join(log_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
            
        return log_dir, log_file

    def log_print(self, message, print_console=True):
        """统一的日志输出函数"""
        if print_console:
            print(message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    @staticmethod
    def parse_options(parser):
        # 基础参数
        parser.add_argument(
            "--file_dir",
            type=str,
            default="data/DBP15K/zh_en",
            required=False,
            help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')",
        )
        parser.add_argument("--rate", type=float, default=0.3, help="training set rate")
        parser.add_argument(
            "--cuda",
            action="store_true",
            default=True,
            help="whether to use cuda or not",
        )
        parser.add_argument("--seed", type=int, default=2021, help="random seed")
        parser.add_argument(
            "--epochs", type=int, default=1000, help="number of epochs to train"
        )
        parser.add_argument("--check_point", type=int, default=100, help="check point")
        
        # 模型架构参数
        parser.add_argument(
            "--hidden_units",
            type=str,
            default="128,128,128",
            help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma",
        )
        parser.add_argument(
            "--heads",
            type=str,
            default="2,2",
            help="heads in each gat layer, splitted with comma",
        )
        parser.add_argument(
            "--instance_normalization",
            action="store_true",
            default=False,
            help="enable instance normalization",
        )
        parser.add_argument(
            "--structure_encoder",
            type=str,
            default="gat",
            choices=["gcn", "gat"],  # 移除了gnn_enhanced选项
            help="the encoder of structure view, [gcn|gat]",
        )
        
        # 训练参数
        parser.add_argument(
            "--lr", type=float, default=0.005, help="initial learning rate"
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-2,
            help="weight decay (L2 loss on parameters)",
        )
        parser.add_argument(
            "--dropout", type=float, default=0.0, help="dropout rate for layers"
        )
        parser.add_argument(
            "--attn_dropout",
            type=float,
            default=0.0,
            help="dropout rate for gat layers",
        )
        parser.add_argument("--bsize", type=int, default=7500, help="batch size")
        
        # 损失函数参数
        parser.add_argument(
            "--tau",
            type=float,
            default=0.1,
            help="the temperature factor of contrastive loss",
        )
        parser.add_argument("--ms_alpha", type=float, default=0.1, help="ms scale_pos")
        parser.add_argument("--ms_beta", type=float, default=40.0, help="ms scale_neg")
        parser.add_argument("--ms_base", type=float, default=0.5, help="ms base")
        
        # 特征维度参数
        parser.add_argument(
            "--attr_dim",
            type=int,
            default=100,
            help="the hidden size of attr and rel features",
        )
        parser.add_argument(
            "--img_dim", type=int, default=100, help="the hidden size of img feature"
        )
        parser.add_argument(
            "--name_dim", type=int, default=100, help="the hidden size of name feature"
        )
        parser.add_argument(
            "--char_dim", type=int, default=100, help="the hidden size of char feature"
        )
        parser.add_argument(
            "--joint_dim", type=int, default=600, help="the hidden size of joint embedding"
        )
        
        # 模态开关
        parser.add_argument(
            "--w_gcn", action="store_true", default=True, help="with gcn features"
        )
        parser.add_argument(
            "--no_gcn", action="store_true", default=False, help="disable gcn features"
        )
        parser.add_argument(
            "--w_rel", action="store_true", default=True, help="with rel features"
        )
        parser.add_argument(
            "--no_rel", action="store_true", default=False, help="disable rel features"
        )
        parser.add_argument(
            "--w_attr", action="store_true", default=True, help="with attr features"
        )
        parser.add_argument(
            "--no_attr", action="store_true", default=False, help="disable attr features"
        )
        parser.add_argument(
            "--w_name", action="store_true", default=True, help="with name features"
        )
        parser.add_argument(
            "--no_name", action="store_true", default=False, help="disable name features"
        )
        parser.add_argument(
            "--w_char", action="store_true", default=True, help="with char features"
        )
        parser.add_argument(
            "--no_char", action="store_true", default=False, help="disable char features"
        )
        parser.add_argument(
            "--w_img", action="store_true", default=True, help="with img features"
        )
        parser.add_argument(
            "--no_img", action="store_true", default=False, help="disable img features"
        )
        
        # VIB参数
        parser.add_argument(
            "--use_graph_vib", action="store_true", default=False, help="use_graph_vib"
        )
        parser.add_argument(
            "--use_attr_vib", action="store_true", default=False, help="use_attr_vib"
        )
        parser.add_argument(
            "--use_img_vib", action="store_true", default=False, help="use_img_vib"
        )
        parser.add_argument(
            "--use_rel_vib", action="store_true", default=False, help="use_rel_vib"
        )
        parser.add_argument(
            "--use_joint_vib", action="store_true", default=False, help="use_joint_vib"
        )
        
        # VIB权重参数
        parser.add_argument("--Beta_g", type=float, default=0.001, help="graph beta")
        parser.add_argument("--Beta_i", type=float, default=0.001, help="image beta")
        parser.add_argument("--Beta_r", type=float, default=0.01, help="relation beta")
        parser.add_argument("--Beta_a", type=float, default=0.001, help="attribute beta")
        parser.add_argument("--joint_beta", type=float, default=1.0, help="joint_beta")
        
        # 其他训练参数
        parser.add_argument(
            "--no_diag", action="store_true", default=False, help="GAT use diag"
        )
        parser.add_argument(
            "--use_project_head",
            action="store_true",
            default=False,
            help="use projection head",
        )
        parser.add_argument(
            "--inner_view_num", type=int, default=6, help="the number of inner view"
        )
        parser.add_argument(
            "--with_weight",
            type=int,
            default=1,
            help="Whether to weight the fusion of different modal features",
        )
        parser.add_argument(
            "--fusion_id", type=int, default=1, help="default weight fusion"
        )
        parser.add_argument(
            "--fusion_strategy", 
            type=str, 
            default="weighted_concat",
            choices=["weighted_concat", "attention", "simple_concat"],
            help="fusion strategy for multimodal features"
        )
        
        # 学习率调度
        parser.add_argument(
            "--use_sheduler", action="store_true", default=False, help="use_sheduler"
        )
        parser.add_argument(
            "--sheduler_gamma", type=float, default=0.98, help="sheduler_gamma"
        )
        parser.add_argument(
            "--use_sheduler_cos",
            action="store_true",
            default=False,
            help="use_sheduler_cos",
        )
        parser.add_argument(
            "--num_warmup_steps", type=int, default=200, help="num_warmup_steps"
        )
        parser.add_argument(
            "--num_training_steps", type=int, default=1000, help="num_training_steps"
        )
        
        # 评估参数
        parser.add_argument(
            "--dist", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')"
        )
        parser.add_argument(
            "--csls", action="store_true", default=False, help="use CSLS for inference"
        )
        parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
        
        # 迭代学习参数
        parser.add_argument(
            "--il", action="store_true", default=False, help="Iterative learning?"
        )
        parser.add_argument(
            "--semi_learn_step",
            type=int,
            default=10,
            help="If IL, what's the update step?",
        )
        parser.add_argument(
            "--il_start", type=int, default=500, help="If Il, when to start?"
        )
        
        # === CLIP相关参数 ===
        parser.add_argument(
            "--use_clip", 
            action="store_true", 
            default=False, 
            help="使用CLIP模型增强特征提取"
        )
        parser.add_argument(
            "--clip_model_name", 
            type=str, 
            default="openai/clip-vit-base-patch32",
            help="CLIP模型名称"
        )
        parser.add_argument(
            "--clip_freeze", 
            action="store_true", 
            default=True, 
            help="是否冻结CLIP参数"
        )
        parser.add_argument(
            "--clip_temperature", 
            type=float, 
            default=0.07, 
            help="CLIP对比学习温度参数"
        )
        parser.add_argument(
            "--clip_weight", 
            type=float, 
            default=0.1, 
            help="CLIP损失权重"
        )
        parser.add_argument(
            "--use_entity_names", 
            action="store_true", 
            default=False, 
            help="使用实体名称进行CLIP文本编码"
        )
        
        # 数据相关参数
        parser.add_argument(
            "--use_mean_img", action="store_true", default=False, help="use_mean_img"
        )
        parser.add_argument(
            "--img_dp_ratio", type=float, default=1.0, help="image dropout ratio"
        )
        parser.add_argument(
            "--word_embedding",
            type=str,
            default="glove",
            help="the type of word embedding, [glove|fasttext]",
        )
        
        # 保存参数
        parser.add_argument(
            "--save_path", type=str, default="save_pkl", help="save path"
        )
        parser.add_argument(
            "--pred_name", type=str, default="pred.txt", help="pred name"
        )
        
        # 其他参数
        parser.add_argument(
            "--alpha", type=float, default=0.2, help="the margin of InfoMaxNCE loss"
        )
        parser.add_argument(
            "--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss"
        )
        parser.add_argument(
            "--zoom", type=float, default=0.1, help="narrow the range of losses"
        )
        
        # 高级训练参数
        parser.add_argument(
            "--use_adaptive_weighting", 
            action="store_true", 
            default=False, 
            help="use adaptive loss weighting"
        )
        parser.add_argument(
            "--use_comprehensive_loss",
            action="store_true",
            default=False,
            help="use comprehensive loss function"
        )
        
        # 多模态融合改进参数
        parser.add_argument(
            "--use_cross_modal_attention",
            action="store_true",
            default=True,
            help="use cross-modal attention in fusion"
        )
        parser.add_argument(
            "--hard_negative_weight",
            type=float,
            default=0.5,
            help="weight for hard negative mining loss"
        )
        parser.add_argument(
            "--cross_modal_weight",
            type=float,
            default=0.3,
            help="weight for cross-modal consistency loss"
        )
        
        args = parser.parse_args()
        
        # 处理 no_xxx 参数覆盖 w_xxx
        if hasattr(args, 'no_gcn') and args.no_gcn:
            args.w_gcn = False
        if hasattr(args, 'no_rel') and args.no_rel:
            args.w_rel = False
        if hasattr(args, 'no_attr') and args.no_attr:
            args.w_attr = False
        if hasattr(args, 'no_name') and args.no_name:
            args.w_name = False
        if hasattr(args, 'no_char') and args.no_char:
            args.w_char = False
        if hasattr(args, 'no_img') and args.no_img:
            args.w_img = False
            
        return args

    @staticmethod
    def set_seed(seed, cuda=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def print_summary(self):
        summary_msg = f"""-----dataset summary-----
dataset:\t{self.args.file_dir}
triple num:\t{len(self.triples)}
entity num:\t{self.ENT_NUM}
relation num:\t{self.REL_NUM}
train ill num:\t{self.train_ill.shape[0]}\ttest ill num:\t{self.test_ill.shape[0]}
CLIP enabled:\t{self.args.use_clip}
Entity names:\t{self.args.use_entity_names}
-------------------------"""
        self.log_print(summary_msg)

    def init_data(self):
        """初始化数据"""
        # Load basic data
        lang_list = [1, 2]
        file_dir = self.args.file_dir
        device = self.device

        self.ent2id_dict, self.ills, self.triples, self.r_hs, self.r_ts, self.ids = (
            read_raw_data(file_dir, lang_list)
        )
        e1 = os.path.join(file_dir, "ent_ids_1")
        e2 = os.path.join(file_dir, "ent_ids_2")
        self.left_ents = get_ids(e1)
        self.right_ents = get_ids(e2)

        self.ENT_NUM = len(self.ent2id_dict)
        self.REL_NUM = len(self.r_hs)
        self.log_print(f"total ent num: {self.ENT_NUM}, rel num: {self.REL_NUM}")

        np.random.shuffle(self.ills)

        # 加载图像特征
        try:
            if abs(1.0 - self.args.img_dp_ratio) > 1e-3:
                self.img_features = load_img_features_dropout(
                    self.ENT_NUM,
                    file_dir,
                    self.triples,
                    self.args.use_mean_img,
                    self.args.img_dp_ratio,
                )
            else:
                self.img_features = load_img_features(
                    self.ENT_NUM, file_dir, self.triples, self.args.use_mean_img
                )
            self.img_features = F.normalize(torch.Tensor(self.img_features).to(device))
            self.log_print(f"image feature shape: {self.img_features.shape}")
        except Exception as e:
            self.log_print(f"Warning: Failed to load image features: {e}")
            # 创建随机图像特征作为备用
            self.img_features = F.normalize(torch.randn(self.ENT_NUM, 2048).to(device))
            self.log_print(f"Using random image features with shape: {self.img_features.shape}")
        
        # 加载实体文本（用于CLIP）
        if self.args.use_clip and self.args.use_entity_names:
            try:
                self.entity_texts = load_entity_texts(file_dir, self.ent2id_dict)
                self.entity_texts = prepare_clip_text_inputs(self.entity_texts)
                self.log_print(f"entity texts loaded: {len(self.entity_texts)}")
            except Exception as e:
                self.log_print(f"Warning: Failed to load entity texts: {e}")
                self.entity_texts = None
        else:
            self.entity_texts = None
        
        # 加载名称和字符特征
        self.name_features = None
        self.char_features = None
        
        if self.args.w_name:
            try:
                e1 = os.path.join(file_dir, "ent_ids_1")
                e2 = os.path.join(file_dir, "ent_ids_2")
                self.name_features = load_name_features(
                    self.ENT_NUM, [e1, e2], self.ent2id_dict, feature_dim=300
                )
                self.name_features = torch.Tensor(self.name_features).to(device)
                self.log_print(f"name feature shape: {self.name_features.shape}")
            except Exception as e:
                self.log_print(f"Warning: Failed to load name features: {e}")
                self.args.w_name = False
        
        if self.args.w_char:
            try:
                e1 = os.path.join(file_dir, "ent_ids_1")
                e2 = os.path.join(file_dir, "ent_ids_2")
                self.char_features = load_char_features(
                    self.ENT_NUM, [e1, e2], self.ent2id_dict, feature_dim=100
                )
                self.char_features = torch.Tensor(self.char_features).to(device)
                self.log_print(f"char feature shape: {self.char_features.shape}")
            except Exception as e:
                self.log_print(f"Warning: Failed to load char features: {e}")
                self.args.w_char = False

        # 训练/测试集划分
        self.train_ill = np.array(
            self.ills[: int(len(self.ills) // 1 * self.args.rate)], dtype=np.int32
        )
        self.test_ill_ = self.ills[int(len(self.ills) // 1 * self.args.rate) :]
        self.test_ill = np.array(self.test_ill_, dtype=np.int32)

        self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).to(device)
        self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).to(device)

        self.left_non_train = list(
            set(self.left_ents) - set(self.train_ill[:, 0].tolist())
        )
        self.right_non_train = list(
            set(self.right_ents) - set(self.train_ill[:, 1].tolist())
        )

        self.log_print(f"#left entity : {len(self.left_ents)}, #right entity: {len(self.right_ents)}")
        self.log_print(f"#left entity not in train set: {len(self.left_non_train)}, #right entity not in train set: {len(self.right_non_train)}")
        
        # 加载关系特征
        try:
            self.rel_features = load_relation(self.ENT_NUM, self.triples, 1000)
            self.rel_features = torch.Tensor(self.rel_features).to(device)
            self.log_print(f"relation feature shape: {self.rel_features.shape}")
        except Exception as e:
            self.log_print(f"Warning: Failed to load relation features: {e}")
            self.rel_features = torch.zeros(self.ENT_NUM, 1000).to(device)
            self.args.w_rel = False
        
        # 加载属性特征
        try:
            a1 = os.path.join(file_dir, "training_attrs_1")
            a2 = os.path.join(file_dir, "training_attrs_2")
            self.att_features = load_attr(
                [a1, a2], self.ENT_NUM, self.ent2id_dict, 1000
            ) 
            self.att_features = torch.Tensor(self.att_features).to(device)
            self.log_print(f"attribute feature shape: {self.att_features.shape}")
        except Exception as e:
            self.log_print(f"Warning: Failed to load attribute features: {e}")
            self.att_features = torch.zeros(self.ENT_NUM, 1000).to(device)
            self.args.w_attr = False

        # 构建邻接矩阵
        try:
            self.adj = get_adjr(
                self.ENT_NUM, self.triples, norm=True
            ) 
            self.adj = self.adj.to(self.device)
        except Exception as e:
            self.log_print(f"Warning: Failed to load adjacency matrix: {e}")
            # 创建单位矩阵作为备用
            indices = torch.arange(self.ENT_NUM).long()
            indices = torch.stack([indices, indices])
            values = torch.ones(self.ENT_NUM)
            self.adj = torch.sparse.FloatTensor(indices, values, (self.ENT_NUM, self.ENT_NUM)).to(self.device)

        self.log_print("Data initialization completed successfully!")

    def init_model(self):
        """初始化模型"""
        img_dim = self.img_features.shape[1]
        char_dim = (
            self.char_features.shape[1] if self.char_features is not None else 100
        )
        
        # 设置设备信息到args中
        self.args.device = self.device

        # 验证和修正参数
        if not hasattr(self.args, 'structure_encoder'):
            self.args.structure_encoder = "gat"
        
        if self.args.structure_encoder not in ["gcn", "gat"]:
            self.log_print(f"Warning: Unknown structure_encoder {self.args.structure_encoder}, using 'gat'")
            self.args.structure_encoder = "gat"
        
        # 检查CLIP可用性
        if self.args.use_clip and not CLIP_AVAILABLE:
            self.log_print("Warning: CLIP requested but transformers not available, disabling CLIP")
            self.args.use_clip = False
            self.args.use_entity_names = False
    
        # 初始化多模态编码器
        try:
            self.multimodal_encoder = IBMultiModal(
                args=self.args,
                ent_num=self.ENT_NUM,
                img_feature_dim=img_dim,
                char_feature_dim=char_dim,
                use_project_head=self.args.use_project_head,
            ).to(self.device)
        except Exception as e:
            self.log_print(f"Error initializing model: {e}")
            # 使用安全的默认参数重试
            self.args.use_clip = False
            self.args.use_comprehensive_loss = False
            self.args.use_project_head = False
            # 禁用所有VIB
            self.args.use_graph_vib = False
            self.args.use_img_vib = False
            self.args.use_rel_vib = False
            self.args.use_attr_vib = False
            self.args.use_joint_vib = False
            
            self.multimodal_encoder = IBMultiModal(
                args=self.args,
                ent_num=self.ENT_NUM,
                img_feature_dim=img_dim,
                char_feature_dim=char_dim,
                use_project_head=False,
            ).to(self.device)
            
            self.log_print("Model initialized with safe defaults")
    
        # 计算参数数量
        self.params = [{"params": list(self.multimodal_encoder.parameters())}]
        total_params = sum(
            p.numel()
            for p in self.multimodal_encoder.parameters()
            if p.requires_grad
        )
        self.log_print(f"Total trainable parameters: {total_params}")
    
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.params, lr=self.args.lr, weight_decay=self.args.weight_decay
        )
    
        # 初始化学习率调度器
        if self.args.use_sheduler:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, gamma=self.args.sheduler_gamma
            )
        elif self.args.use_sheduler_cos:
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.args.num_training_steps,
            )
        
        # 初始化损失函数
        ms_alpha = self.args.ms_alpha
        ms_beta = self.args.ms_beta
        ms_base = self.args.ms_base
        self.criterion_ms = MsLoss(
            device=self.device, thresh=ms_base, scale_pos=ms_alpha, scale_neg=ms_beta
        )
        self.criterion_nce = InfoNCE_loss(device=self.device, temperature=self.args.tau)
        
        # 初始化综合损失函数（改进版本）
        if getattr(self.args, 'use_comprehensive_loss', False):
            try:
                from loss import ImprovedComprehensiveLoss
                self.comprehensive_loss = ImprovedComprehensiveLoss(self.args, self.device)
                self.log_print("Using ImprovedComprehensiveLoss with hard negative mining and cross-modal consistency")
            except Exception as e:
                self.log_print(f"Warning: Failed to initialize ImprovedComprehensiveLoss: {e}")
                try:
                    self.comprehensive_loss = ComprehensiveLoss(self.args, self.device)
                except:
                    self.args.use_comprehensive_loss = False
        
        # 初始化CLIP损失
        if self.args.use_clip:
            try:
                self.clip_loss = CLIPAwareContrastiveLoss(
                    device=self.device,
                    temperature=self.args.clip_temperature,
                    clip_weight=self.args.clip_weight,
                    entity_weight=1.0
                )
            except Exception as e:
                self.log_print(f"Warning: Failed to initialize CLIP loss: {e}")
                self.args.use_clip = False
        
        self.log_print("Model initialization completed successfully!")

    def semi_supervised_learning(self):
        """半监督学习"""
        with torch.no_grad():
            gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = (
                self.multimodal_encoder(
                    self.input_idx,
                    self.adj,
                    self.img_features,
                    self.rel_features,
                    self.att_features,
                    self.name_features,
                    self.char_features,
                    entity_texts=self.entity_texts,
                )
            )

            final_emb = F.normalize(joint_emb)
            distance_list = []
            for i in np.arange(0, len(self.left_non_train), 1000):
                d = pairwise_distances(
                    final_emb[self.left_non_train[i : i + 1000]],
                    final_emb[self.right_non_train],
                )
                distance_list.append(d)
            distance = torch.cat(distance_list, dim=0)
            preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
            preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
            del distance_list, distance, final_emb
            del gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb
        return preds_l, preds_r

    def train(self):
        """训练主函数"""
        config_msg = "model config is:\n" + "\n".join([f"  {k}: {v}" for k, v in vars(self.args).items()])
        self.log_print(config_msg)
        self.log_print("[start training...] ")
        t_total = time.time()
        new_links = []
    
        bsize = self.args.bsize
        device = self.device
    
        self.input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(device)
    
        # VIB参数
        use_graph_vib = self.args.use_graph_vib
        use_attr_vib = self.args.use_attr_vib
        use_img_vib = self.args.use_img_vib
        use_rel_vib = self.args.use_rel_vib
        use_joint_vib = self.args.use_joint_vib
    
        # Beta参数
        Beta_g = self.args.Beta_g
        Beta_i = self.args.Beta_i
        Beta_r = self.args.Beta_r
        Beta_a = self.args.Beta_a

        for epoch in range(self.args.epochs):
            t_epoch = time.time()
            self.multimodal_encoder.train()
            self.optimizer.zero_grad()
    
            # 前向传播
            try:
                gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = (
                    self.multimodal_encoder(
                        self.input_idx,
                        self.adj,
                        self.img_features,
                        self.rel_features,
                        self.att_features,
                        self.name_features,
                        self.char_features,
                        entity_texts=self.entity_texts,
                    )
                )
            except Exception as e:
                self.log_print(f"Forward pass failed at epoch {epoch}: {e}")
                continue
            
            np.random.shuffle(self.train_ill)
            
            for si in np.arange(0, self.train_ill.shape[0], bsize):
                current_links = self.train_ill[si : si + bsize]
                
                if getattr(self.args, 'use_comprehensive_loss', False) and self.comprehensive_loss is not None:
                    # 使用综合损失函数
                    try:
                        embeddings_dict = {
                            'graph': gph_emb,
                            'image': img_emb,
                            'relation': rel_emb,
                            'attribute': att_emb,
                            'name': name_emb,
                            'char': char_emb,
                            'joint': joint_emb
                        }
                        loss_all, loss_components = self.comprehensive_loss(
                            embeddings_dict, current_links, self.multimodal_encoder
                        )
                        
                        # 记录各组件损失
                        epoch_info = f"[epoch {epoch:4d}]"
                        for component, loss_val in loss_components.items():
                            epoch_info += f" {component}:{loss_val:.4f}"
                    except Exception as e:
                        self.log_print(f"Comprehensive loss failed: {e}")
                        loss_all = torch.tensor(0.001, device=device, requires_grad=True)
                        epoch_info = f"[epoch {epoch:4d}] comprehensive_loss_failed"
                    
                else:
                    # 使用传统的分别计算损失方式
                    loss_all = 0
                    epoch_info = f"[epoch {epoch:4d}]"
                    loss_list = []
                    
                    # === 图结构损失 ===
                    if self.args.w_gcn and gph_emb is not None:
                        try:
                            if use_graph_vib and hasattr(self.multimodal_encoder, 'kld_loss'):
                                gph_bce_loss = self.criterion_nce(gph_emb, current_links)
                                gph_kld_loss = self.multimodal_encoder.kld_loss
                                loss_G = gph_bce_loss + Beta_g * gph_kld_loss
                                epoch_info += f" gph_bce:{gph_bce_loss:.4f} gph_kld:{gph_kld_loss:.4f}"
                                loss_list.append(loss_G)
                            else:
                                gph_loss = self.criterion_nce(gph_emb, current_links)
                                epoch_info += f" gph:{gph_loss:.4f}"
                                loss_list.append(gph_loss)
                        except Exception as e:
                            epoch_info += f" [gph_loss_failed]"
    
                    # === 图像损失 ===
                    if self.args.w_img and img_emb is not None:
                        try:
                            if use_img_vib and hasattr(self.multimodal_encoder, 'img_kld_loss'):
                                img_bce_loss = self.criterion_nce(img_emb, current_links)
                                img_kld_loss = self.multimodal_encoder.img_kld_loss
                                loss_I = img_bce_loss + Beta_i * img_kld_loss
                                epoch_info += f" img_bce:{img_bce_loss:.4f} img_kld:{img_kld_loss:.4f}"
                                loss_list.append(loss_I)
                            else:
                                img_loss = self.criterion_nce(img_emb, current_links)
                                epoch_info += f" img:{img_loss:.4f}"
                                loss_list.append(img_loss)
                        except Exception as e:
                            epoch_info += f" [img_loss_failed]"
    
                    # === 关系损失 ===
                    if self.args.w_rel and rel_emb is not None:
                        try:
                            if use_rel_vib and hasattr(self.multimodal_encoder, 'rel_kld_loss'):
                                rel_bce_loss = self.criterion_nce(rel_emb, current_links)
                                rel_kld_loss = self.multimodal_encoder.rel_kld_loss
                                loss_R = rel_bce_loss + Beta_r * rel_kld_loss
                                epoch_info += f" rel_bce:{rel_bce_loss:.4f} rel_kld:{rel_kld_loss:.4f}"
                                loss_list.append(loss_R)
                            else:
                                rel_loss = self.criterion_nce(rel_emb, current_links)
                                epoch_info += f" rel:{rel_loss:.4f}"
                                loss_list.append(rel_loss)
                        except Exception as e:
                            epoch_info += f" [rel_loss_failed]"
    
                    # === 属性损失 ===
                    if self.args.w_attr and att_emb is not None:
                        try:
                            if use_attr_vib and hasattr(self.multimodal_encoder, 'attr_kld_loss'):
                                attr_bce_loss = self.criterion_nce(att_emb, current_links)
                                attr_kld_loss = self.multimodal_encoder.attr_kld_loss
                                loss_A = attr_bce_loss + Beta_a * attr_kld_loss
                                epoch_info += f" attr_bce:{attr_bce_loss:.4f} attr_kld:{attr_kld_loss:.4f}"
                                loss_list.append(loss_A)
                            else:
                                attr_loss = self.criterion_nce(att_emb, current_links)
                                epoch_info += f" attr:{attr_loss:.4f}"
                                loss_list.append(attr_loss)
                        except Exception as e:
                            epoch_info += f" [attr_loss_failed]"
    
                    # === CLIP损失 ===
                    if self.args.use_clip and hasattr(self, 'clip_loss') and hasattr(self.multimodal_encoder, 'clip_features'):
                        try:
                            embeddings_dict = {
                                'graph': gph_emb,
                                'image': img_emb,
                                'relation': rel_emb,
                                'attribute': att_emb,
                                'joint': joint_emb
                            }
                            clip_loss = self.clip_loss(
                                joint_emb,
                                current_links,
                                self.multimodal_encoder.clip_features,
                                embeddings_dict
                            )
                            epoch_info += f" clip:{clip_loss:.4f}"
                            loss_list.append(clip_loss)
                        except Exception as e:
                            epoch_info += f" [clip_loss_failed]"
    
                    # === 联合损失 ===
                    if joint_emb is not None:
                        try:
                            if use_joint_vib and hasattr(self.multimodal_encoder, 'joint_kld_loss'):
                                joint_bce_loss = self.criterion_nce(joint_emb, current_links)
                                joint_kld_loss = self.multimodal_encoder.joint_kld_loss
                                loss_J = joint_bce_loss + self.args.joint_beta * joint_kld_loss
                                epoch_info += f" joint_bce:{joint_bce_loss:.4f} joint_kld:{joint_kld_loss:.4f}"
                                loss_list.append(loss_J)
                            else:
                                joint_ms_loss = self.criterion_ms(joint_emb, current_links)
                                loss_J = joint_ms_loss * self.args.joint_beta
                                epoch_info += f" joint_ms:{joint_ms_loss:.4f}"
                                loss_list.append(loss_J)
                        except Exception as e:
                            epoch_info += f" [joint_loss_failed]"
                                
                    # === 计算总损失 ===
                    if loss_list:
                        loss_all = sum(loss_list)
                    else:
                        # 如果没有有效损失，创建一个小的dummy损失以保持训练继续
                        epoch_info += " [no_valid_loss, using_dummy]"
                        loss_all = torch.tensor(0.001, device=device, requires_grad=True)
                
                epoch_info += f" loss_all:{loss_all:.4f}"
                
                # 只在每个epoch的第一个batch记录详细信息，避免日志过多
                if si == 0:
                    self.log_print(epoch_info, print_console=True)
                else:
                    self.log_print(epoch_info, print_console=False)
                
                # 反向传播
                if loss_all > 0:
                    try:
                        loss_all.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(
                            parameters=self.multimodal_encoder.parameters(),
                            max_norm=0.1,
                            norm_type=2,
                        )
                    except Exception as e:
                        self.log_print(f"Warning: Backward pass failed: {e}", print_console=False)
    
            self.optimizer.step()
            
            # 学习率调度
            if self.args.use_sheduler and epoch > 400:
                self.scheduler.step()
            if self.args.use_sheduler_cos:
                self.scheduler.step()
                
            # 迭代学习
            if (
                epoch >= self.args.il_start
                and (epoch + 1) % self.args.semi_learn_step == 0
                and self.args.il
            ):
                preds_l, preds_r = self.semi_supervised_learning()
                if (epoch + 1) % (
                    self.args.semi_learn_step * 10
                ) == self.args.semi_learn_step:
                    new_links = [
                        (self.left_non_train[i], self.right_non_train[p])
                        for i, p in enumerate(preds_l)
                        if preds_r[p] == i
                    ]
                else:
                    new_links = [
                        (self.left_non_train[i], self.right_non_train[p])
                        for i, p in enumerate(preds_l)
                        if (preds_r[p] == i)
                        and (
                            (self.left_non_train[i], self.right_non_train[p])
                            in new_links
                        )
                    ]
                self.log_print(
                    f"[epoch {epoch}] #links in candidate set: {len(new_links)}"
                )

            if (
                epoch >= self.args.il_start
                and (epoch + 1) % (self.args.semi_learn_step * 10) == 0
                and len(new_links) != 0
                and self.args.il
            ):
                new_links_elect = new_links
                self.log_print(f"\n#new_links_elect: {len(new_links_elect)}")

                self.train_ill = np.vstack((self.train_ill, np.array(new_links_elect)))
                self.log_print(f"train_ill.shape: {self.train_ill.shape}")

                num_true = len([nl for nl in new_links_elect if nl in self.test_ill_])
                self.log_print(f"#true_links: {num_true}")
                self.log_print(
                    f"true link ratio: {100 * num_true / len(new_links_elect):.1f}%"
                )
                for nl in new_links_elect:
                    self.left_non_train.remove(nl[0])
                    self.right_non_train.remove(nl[1])
                self.log_print(
                    f"#entity not in train set: {len(self.left_non_train)} (left) {len(self.right_non_train)} (right)"
                )

                new_links = []
                
            # 检查点评估
            if (epoch + 1) % self.args.check_point == 0:
                self.log_print(f"\n[epoch {epoch:4d}] checkpoint!")
                self.test(epoch)
                
            # 清理内存
            del joint_emb, gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb
            torch.cuda.empty_cache()
    
        self.log_print("[optimization finished!]")
        self.log_print(f"best epoch is {self.best_epoch}, hits@1 hits@10 MRR MR is: {self.best_data_list}")
        self.log_print(f"[total time elapsed: {time.time() - t_total:.4f} s]")
        
        # 保存最终结果
        self.save_final_results()

    def test(self, epoch):
        """测试函数"""
        with torch.no_grad():
            t_test = time.time()
            self.multimodal_encoder.eval()

            try:
                gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb, joint_emb = (
                    self.multimodal_encoder(
                        self.input_idx,
                        self.adj,
                        self.img_features,
                        self.rel_features,
                        self.att_features,
                        self.name_features,
                        self.char_features,
                        entity_texts=self.entity_texts,
                    )
                )
            except Exception as e:
                self.log_print(f"Test forward pass failed: {e}")
                return

            final_emb = F.normalize(joint_emb)
            top_k = [1, 10, 50]
            
            acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
            acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
            test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = (
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            
            try:
                if self.args.dist == 2:
                    distance = pairwise_distances(
                        final_emb[self.test_left], final_emb[self.test_right]
                    )
                else:
                    import scipy.spatial as T
                    distance = torch.FloatTensor(
                        T.distance.cdist(
                            final_emb[self.test_left].cpu().data.numpy(),
                            final_emb[self.test_right].cpu().data.numpy(),
                            metric="cityblock",
                        )
                    )

                if self.args.csls is True:
                    distance = 1 - csls_sim(1 - distance, self.args.csls_k)
            except Exception as e:
                self.log_print(f"Distance calculation failed: {e}")
                return

            to_write = []
            test_left_np = self.test_left.cpu().numpy()
            test_right_np = self.test_right.cpu().numpy()
            to_write.append(
                [
                    "idx",
                    "rank",
                    "query_id",
                    "gt_id",
                    "ret1",
                    "ret2",
                    "ret3",
                    "v1",
                    "v2",
                    "v3",
                ]
            )

            for idx in range(self.test_left.shape[0]):
                values, indices = torch.sort(distance[idx, :], descending=False)
                rank = (indices == idx).nonzero().squeeze().item()
                mean_l2r += rank + 1
                mrr_l2r += 1.0 / (rank + 1)
                for i in range(len(top_k)):
                    if rank < top_k[i]:
                        acc_l2r[i] += 1

                indices = indices.cpu().numpy()
                to_write.append(
                    [
                        idx,
                        rank,
                        test_left_np[idx],
                        test_right_np[idx],
                        test_right_np[indices[0]],
                        test_right_np[indices[1]],
                        test_right_np[indices[2]],
                        round(values[0].item(), 4),
                        round(values[1].item(), 4),
                        round(values[2].item(), 4),
                    ]
                )

            for idx in range(self.test_right.shape[0]):
                _, indices = torch.sort(distance[:, idx], descending=False)
                rank = (indices == idx).nonzero().squeeze().item()
                mean_r2l += rank + 1
                mrr_r2l += 1.0 / (rank + 1)
                for i in range(len(top_k)):
                    if rank < top_k[i]:
                        acc_r2l[i] += 1

            mean_l2r /= self.test_left.size(0)
            mean_r2l /= self.test_right.size(0)
            mrr_l2r /= self.test_left.size(0)
            mrr_r2l /= self.test_right.size(0)
            for i in range(len(top_k)):
                acc_l2r[i] = round(acc_l2r[i] / self.test_left.size(0), 4)
                acc_r2l[i] = round(acc_r2l[i] / self.test_right.size(0), 4)
            
            # 记录结果
            result = {
                'epoch': epoch,
                'l2r_h1': float(acc_l2r[0]),
                'l2r_h10': float(acc_l2r[1]),
                'l2r_mrr': float(mrr_l2r),
                'l2r_mr': float(mean_l2r),
                'r2l_h1': float(acc_r2l[0]),
                'r2l_h10': float(acc_r2l[1]),
                'r2l_mrr': float(mrr_r2l),
                'r2l_mr': float(mean_r2l),
                'time': time.time() - t_test
            }
            self.exp_results.append(result)
                
            del (
                distance,
                gph_emb,
                img_emb,
                rel_emb,
                att_emb,
                name_emb,
                char_emb,
                joint_emb,
            )
            gc.collect()
            
            # 输出测试结果（格式简洁）
            test_msg = f"epoch {epoch:4d} | l2r: h1={acc_l2r[0]:.4f} h10={acc_l2r[1]:.4f} mrr={mrr_l2r:.4f} mr={mean_l2r:.2f} | r2l: h1={acc_r2l[0]:.4f} h10={acc_r2l[1]:.4f} mrr={mrr_r2l:.4f} mr={mean_r2l:.2f} | t={time.time()-t_test:.2f}s"
            self.log_print(test_msg)
            
            if acc_l2r[0] > self.best_hit_1:
                self.best_hit_1 = acc_l2r[0]
                self.best_epoch = epoch
                self.best_data_list = [
                    acc_l2r[0],
                    acc_l2r[1],
                    mrr_l2r,
                    mean_l2r,
                ]
                import copy
                self.best_to_write = copy.deepcopy(to_write)

            if epoch + 1 == self.args.epochs:
                pred_name = self.args.pred_name
                import csv

                save_path = os.path.join(self.args.save_path, pred_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                with open(os.path.join(save_path, pred_name + ".txt"), "w") as f:
                    wr = csv.writer(f, dialect="excel")
                    wr.writerows(self.best_to_write)

    def save_final_results(self):
        """保存最终结果 - 修复JSON序列化错误"""
        
        # 转换numpy类型为Python原生类型
        def convert_to_python_type(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, list):
                return [convert_to_python_type(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_python_type(v) for k, v in obj.items()}
            else:
                return obj
        
        results = {
            'config': {k: v for k, v in vars(self.args).items() 
                      if isinstance(v, (str, int, float, bool))},
            'dataset_info': {
                'entity_num': int(self.ENT_NUM),
                'relation_num': int(self.REL_NUM),
                'train_ill_num': int(len(self.train_ill)),
                'test_ill_num': int(len(self.test_ill))
            },
            'best_results': {
                'epoch': int(self.best_epoch),
                'hit_1': float(self.best_data_list[0]) if self.best_data_list else 0.0,
                'hit_10': float(self.best_data_list[1]) if len(self.best_data_list) > 1 else 0.0,
                'mrr': float(self.best_data_list[2]) if len(self.best_data_list) > 2 else 0.0,
                'mr': float(self.best_data_list[3]) if len(self.best_data_list) > 3 else 0.0
            },
            'all_results': convert_to_python_type(self.exp_results)
        }
        
        try:
            with open(os.path.join(self.log_dir, "results.json"), 'w') as f:
                json.dump(results, f, indent=2)
            
            final_msg = f"\nResults saved to: {self.log_dir}"
            self.log_print(final_msg)
        except Exception as e:
            self.log_print(f"Warning: Failed to save results: {e}")
            # 保存简化版本
            simple_results = {
                'best_hit1': float(self.best_data_list[0]) if self.best_data_list else 0.0,
                'best_epoch': int(self.best_epoch)
            }
            with open(os.path.join(self.log_dir, "simple_results.json"), 'w') as f:
                json.dump(simple_results, f, indent=2)


if __name__ == "__main__":
    model = KACEA()
    model.train()