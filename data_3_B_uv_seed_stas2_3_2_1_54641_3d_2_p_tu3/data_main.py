import numpy as np
import torch
import pandas as pd
from train_3_B import train_and_evaluate_from_npy
def to_array(x):
    if isinstance(x, list):
        try:
            return np.stack([np.asarray(e) for e in x], axis=0)
        except Exception:
            return np.asarray(x)
    return np.asarray(x)
def ensure_5d(a):
    """
    目标统一为 [B, T, C, H, W]
    - [T, H, W]        -> [1, T, 1, H, W]
    - [B, T, H, W]     -> [B, T, 1, H, W]
    - [B, T, C, H, W]  -> 原样
    其他维度直接报错，避免误判。
    """
    a = np.asarray(a)
    if a.ndim == 5:
        return a
    if a.ndim == 4:
        # 假设 [B, T, H, W]
        B, T, H, W = a.shape
        return a.reshape(B, T, 1, H, W)
    if a.ndim == 3:
        # [T, H, W]
        T, H, W = a.shape
        return a.reshape(1, T, 1, H, W)
    raise ValueError(f"Unsupported model shape {a.shape}")
def ensure_BT(y):
    """
    目标统一为 [B, T]
    - [T]        -> [1, T]
    - [B, T]     -> 原样
    - [B, T, 1]  -> [B, T]
    """
    y = np.asarray(y)
    if y.ndim == 2:
        return y
    if y.ndim == 1:
        T = y.shape[0]
        return y.reshape(1, T)
    if y.ndim == 3 and y.shape[-1] == 1:
        return y[..., 0]
    raise ValueError(f"Unsupported labels shape {y.shape}")

# def load_all_npy(time_list):
#     """
#     读取多个起报时的数据，统一到:
#       model_train_all: [B, T, C(+1), H, W]  (多了1个起报时通道)
#       obs_train_all  : [B, T]
#       model_val_all  : [B, T, C(+1), H, W]
#       obs_val_all    : [B, T]
#     """
#     hour_encoding = {t: idx for idx, t in enumerate(time_list)}
#     num_init_times = len(time_list)

#     all_train_model, all_train_obs = [], []
#     all_val_model,   all_val_obs   = [], []
#     for ti in time_list:
#         train_u_file = f"/thfs1/home/qx_hyt/hpp/data/station_AI/train_data/train_data_u_{ti}.npy"
#         train_v_file = f"/thfs1/home/qx_hyt/hpp/data/station_AI/train_data/train_data_v_{ti}.npy"
#         val_u_file   = f"/thfs1/home/qx_hyt/hpp/data/station_AI/train_data/val_data_u_{ti}.npy"
#         val_v_file   = f"/thfs1/home/qx_hyt/hpp/data/station_AI/train_data/val_data_v_{ti}.npy"
        
        
#         train_obs_file   = f"/thfs1/home/qx_hyt/hpp/data/station_AI/train_data/train_labels_{ti}.npy"
#         val_obs_file     = f"/thfs1/home/qx_hyt/hpp/data/station_AI/train_data/val_labels_{ti}.npy"

#         try:
#             train_u = np.load(train_u_file, allow_pickle=True)
#             train_v = np.load(train_v_file, allow_pickle=True)
#             val_u   = np.load(val_u_file,   allow_pickle=True)
#             val_v   = np.load(val_v_file,   allow_pickle=True)
#             # train_model = np.load(train_model_file, allow_pickle=True)
#             # val_model   = np.load(val_model_file,   allow_pickle=True)
#             train_obs= np.load(train_obs_file,   allow_pickle=True)
#             val_obs  = np.load(val_obs_file,     allow_pickle=True)


#             train_u   = to_array(train_u).astype(np.float32)
#             train_v   = to_array(train_v).astype(np.float32)
#             val_u     = to_array(val_u).astype(np.float32)
#             val_v     = to_array(val_v).astype(np.float32)
#             train_obs = to_array(train_obs).astype(np.float32)
#             val_obs   = to_array(val_obs).astype(np.float32)

#             # ---- NWP 输入统一到 [B,T,C,H,W] ----
        
#             train_u = ensure_5d(train_u)
#             train_v   = ensure_5d(train_v)
#             val_u = ensure_5d(val_u)
#             val_v   = ensure_5d(val_v)

#             # ---- 标签统一到 [B,T] ----
    
#             # ---- 统一成 ndarray ----
#             train_obs = ensure_BT(train_obs)
#             val_obs   = ensure_BT(val_obs)

#             # ---- 加“起报时通道”到 C 维（每个 ti 一个常数通道） ----
#             code = hour_encoding[ti]
#             hour_norm = np.float32(code / max(1, (num_init_times - 1)))  # 归一化到[0,1]

#             B, T, C, H, W = train_u.shape
#             hour_chan_train = np.full((B, T, 1, H, W), hour_norm, dtype=np.float32)
#             train_model = np.concatenate([train_u, train_v], axis=2)  # C -> C+1
#             train_model = np.concatenate([train_model, hour_chan_train], axis=2)  # C -> C+1

#             Bv, Tv, Cv, Hv, Wv = val_u.shape
#             hour_chan_val = np.full((Bv, Tv, 1, Hv, Wv), hour_norm, dtype=np.float32)
#             val_model = np.concatenate([val_u, val_v], axis=2)        # C -> C+1
#             val_model = np.concatenate([val_model, hour_chan_val], axis=2)        # C -> C+1

#             # ---- 累加 ----
#             all_train_model.append(train_model)
#             all_train_obs.append(train_obs)
#             all_val_model.append(val_model)
#             all_val_obs.append(val_obs)

#             print(f"[{ti}] 加载成功 train:{train_model.shape}, val:{val_model.shape}")
#         except Exception as e:
#             print(f"[{ti}] 加载失败: {e}")

# ---- 用法示例 ----
# time_list = ["02","05","08","11","14","17","20","23"]
# mtr, ytr, mva, yva = load_all_npy(time_list, root=ROOT, vars_to_use=("10u","10v","2DPT"))

    # ---- 拼接所有起报时 ----
    # model_train_all = np.concatenate(all_train_model, axis=0)
    # obs_train_all   = np.concatenate(all_train_obs,   axis=0)
    # model_val_all   = np.concatenate(all_val_model,   axis=0)
    # obs_val_all     = np.concatenate(all_val_obs,     axis=0)

    # print("合并后：")
    # print("  model_train_all:", model_train_all.shape)  # [B,T,C(+1),H,W]
    # print("  obs_train_all  :", obs_train_all.shape)    # [B,T]
    # print("  model_val_all  :", model_val_all.shape)
    # print("  obs_val_all    :", obs_val_all.shape)

    # return model_train_all, obs_train_all, model_val_all, obs_val_all

import os, re, numpy as np
from collections import defaultdict

ROOT = "/thfs1/home/qx_hyt/hpp/data/station_AI/train_data2"  
VARS = ("10u", "10v")                                

_data_re   = re.compile(r'^(train|val)_data_(10u|10v|2DPT)_([A-Za-z]\d{4})_(02)\.npy$')
_label_re  = re.compile(r'^(train|val)_labels_([A-Za-z]\d{4})_(02)\.npy$')

def build_index(root=ROOT):
    data_idx   = defaultdict(dict)   
    labels_idx = defaultdict(dict)  
    for fn in os.listdir(root):
        m = _data_re.match(fn)
        if m:
            split, var, station= m.groups()
            data_idx[(split, station, var)] = os.path.join(root, fn)
            continue
        m = _label_re.match(fn)
        if m:
            split, station= m.groups()
            labels_idx[(split, station)] = os.path.join(root, fn)

    stations = sorted({k[1] for k in data_idx.keys()})
    return data_idx, labels_idx, stations


def load_all_npy(root=ROOT, vars_to_use=VARS):
    
    data_idx, labels_idx, stations = build_index(root)
    # hour_encoding   = {t: i for i, t in enumerate(time_list)}
    # num_init_times  = len(time_list)
    train_models, train_obs, train_meta = [], [], []  
    val_models,   val_obs,   val_meta   = [], [], [] 

    def load_one(split, station):
        arrs = []
        for v in vars_to_use:
            p = data_idx[(split, station, v)]
            a = np.load(p, allow_pickle=True)
            # print(a.shape)
            a = to_array(a).astype(np.float32)
            a = ensure_5d(a)     
            arrs.append(a)
        model = np.concatenate(arrs, axis=2)  

        # 读标签并裁剪到 [Bm, Tm]
        p_lab = labels_idx[(split, station)]
        lab   = np.load(p_lab, allow_pickle=True)
        lab   = to_array(lab).astype(np.float32)
        lab   = ensure_BT(lab)#[:Bm, :Tm]

        return model, lab

    # 遍历 split / station / ti
    for station in stations:
        # for ti in time_list:
        out = load_one("train", station)
        if out is not None:
            m, y = out
            train_models.append(m)
            train_obs.append(y)
            train_meta.append((station, m.shape[0])) 
            print(f"[train][{station}]02 载入成功 model{m.shape}, obs{y.shape}")
        out = load_one("val", station)
        if out is not None:
            m, y = out
            val_models.append(m)
            val_obs.append(y)
            val_meta.append((station, m.shape[0]))  
            print(f"[val  ][{station}]02 载入成功 model{m.shape}, obs{y.shape}")

    # 合并
    if not train_models and not val_models:
        raise RuntimeError("没有成功载入的 train/val 数据，请检查缺失提示。")

    # 统计各自 split 的 T 列表
    train_Ts = [m.shape[1] for m in train_models]
    val_Ts   = [m.shape[1] for m in val_models]
    Tmin_tr  = min(train_Ts) if train_Ts else None
    Tmin_va  = min(val_Ts)   if val_Ts   else None

    # 可选：打印一下，帮助自检
    if Tmin_tr is not None:
        print(f"[train] 各样本 T 范围: min={min(train_Ts)}, max={max(train_Ts)}，统一裁剪到 T={Tmin_tr}")
    if Tmin_va is not None:
        print(f"[val  ] 各样本 T 范围: min={min(val_Ts)},   max={max(val_Ts)}，统一裁剪到 T={Tmin_va}")

# 按 Tmin 裁剪（同时裁剪标签），然后再拼
    if train_models:
        # train_models = [m[:, :Tmin_tr] for m in train_models]        # [B, Tmin, C, H, W]
        # train_obs    = [y[:, :Tmin_tr] for y in train_obs]           # [B, Tmin]
        model_train_all = np.concatenate(train_models, axis=0)
        obs_train_all   = np.concatenate(train_obs,   axis=0)
    else:
        model_train_all, obs_train_all = None, None

    if val_models:
        # val_models = [m[:, :Tmin_va] for m in val_models]
        # val_obs    = [y[:, :Tmin_va] for y in val_obs]
        model_val_all = np.concatenate(val_models, axis=0)
        obs_val_all   = np.concatenate(val_obs,   axis=0)
    else:
        model_val_all, obs_val_all = None, None

    print("合并后：")
    print("model_train_all:", None if model_train_all is None else model_train_all.shape)
    print("obs_train_all  :", None if obs_train_all   is None else obs_train_all.shape)
    print("model_val_all  :", None if model_val_all   is None else model_val_all.shape)
    print("obs_val_all    :", None if obs_val_all     is None else obs_val_all.shape)

    return model_train_all, obs_train_all, model_val_all, obs_val_all, train_meta, val_meta

def build_station_lookup(csv_path):
    df = pd.read_csv(csv_path, dtype={'Station_Id_C': str}, low_memory=False)
    df['Station_Id_C'] = df['Station_Id_C'].str.strip().str.upper()
    elev_col = 'Alti' if 'Alti' in df.columns else ('Alt' if 'Alt' in df.columns else None)
    dfu = (df.drop_duplicates('Station_Id_C', keep='last')
             .set_index('Station_Id_C')[['Lat','Lon',elev_col]])
    return dfu.apply(lambda r: (float(r['Lat']), float(r['Lon']), float(r[elev_col])), axis=1).to_dict()

def coords_from_meta(meta_list, station_lut):
    blocks = []
    for sid, bcount in meta_list:
        key = str(sid).strip().upper()
        if key not in station_lut:
            raise KeyError(f"站点 {key} 不在站点表里")
        lat, lon, elev = station_lut[key]
        blocks.append(np.repeat([[lat, lon, elev]], bcount, axis=0).astype(np.float32))
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0,3), np.float32)

if __name__ == "__main__":
    # time_list = ['02', '05', '08', '11', '14', '17', '20', '23']
    csv_path='/thfs1/home/qx_hyt/hpp/data/station_AI/2023.csv'
    model_train, obs_train, model_val, obs_val, meta_tr, meta_va  = load_all_npy(root=ROOT, vars_to_use=VARS)

    lut        = build_station_lookup(csv_path)
    coords_tr  = coords_from_meta(meta_tr, lut)   # [B_tr, 3]
    coords_va  = coords_from_meta(meta_va, lut)   # [B_va, 3]


    print("\n========== 开始统一训练模型 ==========\n")

    # 你的训练函数应接受四个输入（不需要 hour_codes）
    model = train_and_evaluate_from_npy(
        model_train, obs_train,
        model_val,  obs_val,
        coords_train=coords_tr,   
        coords_val=coords_va,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
