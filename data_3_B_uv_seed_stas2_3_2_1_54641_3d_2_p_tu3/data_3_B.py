import numpy as np
import pandas as pd
# def load_station_info(csv_path):
#     df = pd.read_csv(csv_path)
#     df['Station_Id_C'] = df['Station_Id_C'].astype(str).str.strip()
#     dfu = (df.drop_duplicates('Station_Id_C', keep='last')
#              .set_index('Station_Id_C')[['Lat','Lon','Alti']])
#     return dfu  # index=站号(str)，列为 Lat/Lon/Alti

# station_df = load_station_info('/mnt/d/hpp_onedrive/OneDrive/work/data/station2/2023.csv')
# station_id = ['A2662','A3171','54630','54641','54538','54646','54741','54647','54544','54558',
#                   'L2232','54552','54456','L2207','D3118','54649','54750','D2002','54660','54661',
#                   '54655','54659','54579','D1104','D0029','D0046']
# def station_coords(sta_id: str):
#     key = str(sta_id).strip()
#     r = station_df.loc[key]
#     return float(r['Lat']), float(r['Lon']), float(r['Alti'])

# def build_dataset(model_speed, obs_speed, init_hour, history_hours=24, forecast_hours=24):
#     hist_obs_list = []
#     nwp_seq_list = []
#     coords_list = []
#     y_list = []
#     # mean_list = []
#     # last_obs_list = []

#     num_samples = len(obs_speed) - history_hours - forecast_hours

#     for t in range(num_samples): 
#         hist_obs = obs_speed[t : t + history_hours]  # [history_hours, ...]
#         # hist_obs = hist_obs[:, None]  
#         nwp_seq = model_speed[t + history_hours : t + history_hours + forecast_hours]
#         # nwp_seq = nwp_seq[:, None, :, :] 
#         y_future = obs_speed[t + history_hours : t + history_hours + forecast_hours]

#         coords = np.array([lat, lon], dtype=np.float32)  
#         # coords[:, 0] = lat
#         # coords[:, 1] = lon
#         # 取模式均值
#         # mean_val = np.mean(model_speed[t + history_hours : t + history_hours + forecast_hours])

#         hist_obs_list.append(hist_obs)
#         nwp_seq_list.append(nwp_seq)
#         coords_list.append(coords)
#         y_list.append(y_future)
#         # mean_list.append(mean_val)
#         # 历史最后一个观测值
#         # last_obs_list.append(hist_obs[-1])

#     # 转成 numpy 数组
#     hist_obs_arr = np.array(hist_obs_list)
#     nwp_seq_arr = np.array(nwp_seq_list)
#     coords_arr = np.array(coords_list)
#     y_arr = np.array(y_list)
#     # mean_arr = np.array(mean_list)
#     # last_obs_arr = np.array(last_obs_list)

#     return hist_obs_arr, nwp_seq_arr, coords_arr, y_arr
# station_df = load_station_info('/mnt/d/hpp_onedrive/OneDrive/work/data/station2/2023.csv')
# csv_path='/thfs1/home/qx_hyt/hpp/data/station_AI/2023.csv'
# station_id = ['A2662','A3171']
# station_id = ['A2662','A3171','54630','54641','54538','54646','54741','54647','54544','54558',
#                   'L2232','54552','54456','L2207','D3118','54649','54750','D2002','54660','54661',
#                   '54655','54659','54579','D1104','D0029','D0046']
# time_list = ['02', '05', '08', '11', '14', '17', '20', '23']

# def coords_in_given_order(station_ids, csv_path):
#     df = pd.read_csv(csv_path)
#     df['Station_Id_C'] = df['Station_Id_C'].astype(str).str.strip()
#     dfu = (df.drop_duplicates('Station_Id_C', keep='last')
#              .set_index('Station_Id_C')[['Lat','Lon','Alti']])
#     out = np.zeros((len(station_ids), 3), dtype=np.float32)
#     for i, sid in enumerate(station_ids):
#         r = dfu.loc[str(sid).strip()]
#         out[i] = [float(r['Lat']), float(r['Lon']), float(r['Alti'])]
#     return out 

# coords_station = coords_in_given_order(station_id, csv_path)  # [S,3]
# coords_per_b = np.repeat(coords_station, repeats=len(time_list), axis=0)  # [S*len(time_list), 3]
# print(coords_per_b.shape)

# def build_dataset_batched(model_all,         # [B, T, C, H, W]
#                           obs_all,           # [B, T]
#                         #   coords_per_b=None, 
#                           history_hours=24,
#                           forecast_hours=24):
#     T, C, H, W = model_all.shape                    
#     hist_list, nwp_list, coord_list, y_list = [], [], [], []
#     # for b in range(B):
#         # lat, lon, elev = coords_per_b[b].astype(np.float32)
#         # coord_vec = np.array([lat, lon, elev], dtype=np.float32)
#     t_max = T - history_hours - forecast_hours
#     for t0 in range(t_max):
#         t1 = t0 + history_hours
#         t2 = t1 + forecast_hours
#         hist = obs_all[t0:t1].astype(np.float32)                # [H]
#         nwp  = model_all[t1:t2, :, :, :].astype(np.float32)     # [F,C,H,W]
#         y    = obs_all[t1:t2].astype(np.float32)                # [F]
#         # coord= np.array([lat, lon, elev], dtype=np.float32)        # [3]
#         hist_list.append(hist) 
#         nwp_list.append(nwp)
#         # coord_list.append(coord_vec)
#         y_list.append(y)    

#     hist_obs = np.stack(hist_list,  axis=0)   # [N, H]
#     nwp_seq  = np.stack(nwp_list,   axis=0)   # [N, F, C, H, W]
#     coords   = np.stack(coord_list, axis=0)   # [N, 3]
#     y        = np.stack(y_list,     axis=0)   # [N, F]
#     return hist_obs, nwp_seq, coords, y


def standardize(data):
    mean = data.mean()
    std = data.std()
    norm = (data - mean) / std
    return norm, mean, std
