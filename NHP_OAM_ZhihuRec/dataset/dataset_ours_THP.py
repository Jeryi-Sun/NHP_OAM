import datetime
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from config import const
import numpy as np
import pytz



rec_inter_history = pickle.load(open(const.recommendation_index, 'rb'))
query_inter_history = pickle.load(open(const.query_index, 'rb'))
#open_search_inter_history = pickle.load(open(const.open_search_index, 'rb'))

# search_vocab = pickle.load(open(const.search_vocab,'rb'))
#reco_vocab = pickle.load(open(const.reco_vocab,'rb'))
user_vocab = pickle.load(open(const.user_vocab, 'rb'))
item_vocab = pickle.load(open(const.item_vocab, 'rb'))


def get_whms_features(time):
    beijing_tz = pytz.timezone('Asia/Shanghai')
    # 假设 timestamp 是一个时间戳，比如 1626353460
    timestamp = time

    # 从时间戳创建 UTC 时间的 datetime 对象
    dt_utc = datetime.datetime.utcfromtimestamp(timestamp)

    # 将 UTC 时间转换为北京时间
    dt_beijing = dt_utc.replace(tzinfo=pytz.utc).astimezone(beijing_tz)


    # 提取时、分、秒
    week = dt_beijing.weekday()
    hour = dt_beijing.hour
    minute = dt_beijing.minute
    second = dt_beijing.second
    return [week+1, hour+1, minute+1, second+1]

class MyDataset(Dataset):
    def __init__(self, main_file_path, train_file, valid_file, test_file):
        self.df = pd.read_csv(main_file_path)
        self.train_df = pd.read_csv(train_file)
        self.valid_df = pd.read_csv(valid_file)
        self.test_df = pd.read_csv(test_file)
        self.all_df = pd.concat([self.train_df, self.valid_df, self.test_df], axis=0).reset_index(drop=True)

        self.rec_inter_history = rec_inter_history
        self.query_inter_history = query_inter_history

    def __len__(self):
        return len(self.df)
    def get_time_feature(self, time):
        timestamp = time 
        dt = datetime.datetime.fromtimestamp(timestamp)
        hour_sin = np.sin(2 * np.pi * dt.hour / 24, dtype=np.float32)
        hour_cos = np.cos(2 * np.pi * dt.hour / 24, dtype=np.float32)
        minute_sin = np.sin(2 * np.pi * dt.minute / 60, dtype=np.float32)
        minute_cos = np.cos(2 * np.pi * dt.minute / 60, dtype=np.float32)
        second_sin = np.sin(2 * np.pi * dt.second / 60, dtype=np.float32)
        second_cos = np.cos(2 * np.pi * dt.second / 60, dtype=np.float32)

        time_features = [hour_sin, hour_cos, minute_sin, minute_cos, second_sin, second_cos]

        return time_features        
    
    def get_intervel_features(self, now_time, time):
        inter_time_min = (now_time-time)/60 + 1 # 0是 padding
        return inter_time_min
    
    def __getitem__(self, idx):
        user_id, time, label, rec_pos, query_pos = self.df.loc[idx, ['user_id', 'request_time_ms', 'label', 'rec_pos', 'query_pos']]
        rec_pos, query_pos = int(rec_pos), int(query_pos)
        #rec_inter_history_s = [item_vocab[i] for i in self.rec_inter_history[user_id][0][:rec_pos]]
        #search_inter_history_s = [item_vocab[i] for i in self.search_inter_history[user_id][0][:src_pos]]
        #query_inter_history_s = self.query_inter_history[user_id][0][:query_pos]
        # 选取相同 user_id，且 request_time_ms 小于当前行的数据
        filtered_df = self.all_df[(self.all_df['user_id'] == user_id) & (self.all_df['request_time_ms'] < time)]
        filtered_df = filtered_df.sort_values(by='request_time_ms')  # 按时间排序

        rec_poses = filtered_df['rec_pos'].values.tolist()  # 获取所有 rec_pos
        rec_poses.append(rec_pos)
        query_poses = filtered_df['query_pos'].values.tolist()  # 获取所有 rec_pos
        query_poses.append(query_pos)
        session_time_list = filtered_df['request_time_ms'].values.tolist()  # 获取所有 rec_pos
        session_label = [label + 1 for label in filtered_df['label'].values.tolist()]  # 获取所有的 session_label，需要 padding 所以所有 label 都需要+1
        


        rec_inter_history = self.rec_inter_history[user_id][0][:rec_pos]  # 获取用户交互记录
        rec_inter_history = [item_vocab[i] for i in rec_inter_history]
        rec_inter_time =  self.rec_inter_history[user_id][1][:rec_pos]
        rec_sessions = []  # 初始化 session 列表
        rec_sessions_time = []  # 初始化 session 列表
        


        query_inter_history = self.query_inter_history[user_id][0][:query_pos]  # 获取用户交互记录
        query_inter_time =  self.query_inter_history[user_id][1][:query_pos]
        query_sessions = []  # 初始化 session 列表
        query_sessions_time = []  # 初始化 session 列表


        """
        rec_inter_time query_inter_time search_inter_time session_time_list 均为 Python list 
        1.  输出四个列表里最大的值
        2. 拼接输入到normalization_time函数中，并将返回的列表切分之后还原到四个 list 上
        """


        ## 先对 rec 进行操作
        for i in range(len(rec_poses)):  # 倒序遍历 rec_pos
            # 每两个 rec_pos 之间的 item 构成一个 session
            if i==0:
                session = rec_inter_history[:rec_poses[i]]
            else:
                start = rec_poses[i-1]
                end = rec_poses[i]
                session = rec_inter_history[start:end]
                # 若 session 长度不足 10，前填充 0

            rec_sessions.append(session)
            if i == 0:
                session_time = rec_inter_time[:rec_poses[0]]
            else:
                start = rec_poses[i-1]
                end = rec_poses[i]
                session_time = rec_inter_time[start:end] 

            rec_sessions_time.append(session_time)



        # 再对 query 进行操作
        for i in range(len(query_poses)):  # 倒序遍历 query_pos
            # 每两个 query_pos 之间的 item 构成一个 session

            if i == 0:
                session = query_inter_history[:query_poses[0]]
            else:
                start = query_poses[i-1]
                end = query_poses[i]
                session = query_inter_history[start:end]

            # 若 session 长度不足 10，前填充 0
            query_sessions.append(session)

            if i == 0:
                session_time = query_inter_time[:query_poses[0]]
            else:
                start = query_poses[i-1]
                end = query_poses[i]
                session_time = query_inter_time[start:end]            
            
            query_sessions_time.append(session_time)

        # session time padding
        session_time_list = pad_or_truncate(session_time_list, const.ours_session_len)
        session_label_list = pad_or_truncate(session_label, const.ours_session_len)



        rec_sessions = pad_or_truncate_2d(rec_sessions, const.ours_session_len, const.ours_max_reco_len)
        query_sessions = pad_or_truncate_3d(query_sessions, const.ours_session_len, const.ours_max_query_len, const.ours_max_word_len)

        rec_inter_time = pad_or_truncate_2d(rec_sessions_time, const.ours_session_len, const.ours_max_reco_len)
        query_inter_time = pad_or_truncate_2d(query_sessions_time, const.ours_session_len, const.ours_max_query_len)

        """
        query_sessions_time search_sessions_time rec_sessions_time 均为 2d list
        session_time_list 为 1d list
        拼接之后输入到 normalization_time 函数中，将返回的列表切分之后还原到四个 list 上

        注意需要修改 normalization_time 函数，因为输入的包含 2d  和 1d list，而以下的 normalization_time 函数只能处理 1d list

        """

        # 找到最小非零值
        min_val = find_min_value(query_inter_time, rec_inter_time, session_time_list)

        # 对四个列表进行归一化处理
        query_sessions_time_normalized = normalization_time(query_inter_time, min_val)
        rec_sessions_time_normalized = normalization_time(rec_inter_time, min_val)
        session_time_list_normalized = normalization_time(session_time_list, min_val)
        #time = (time-min_val)/60+1
        lag_time = min(int((time-min_val)/60-rec_sessions_time_normalized[-1][-1]+1), const.ours_lag_time_vocab)


        return rec_sessions, query_sessions,  \
                query_sessions_time_normalized,  rec_sessions_time_normalized, session_time_list_normalized, \
                user_vocab[user_id], time, lag_time, label, session_label_list # session_label_list 这个个人感觉可以加上最后的 label 之后 prediction 不动，label 截取一下，把第一个扔了

def find_min_value(*args):
    min_val = float('inf')
    for arg in args:
        if isinstance(arg[0], list):  # 如果是2D列表
            for lst in arg:
                for val in lst:
                    if val > 0 and val < min_val:
                        min_val = val
        else:  # 如果是1D列表
            for val in arg:
                if val > 0 and val < min_val:
                    min_val = val
    if min_val == float('inf'):
        min_val = 0      
    return min_val

def normalization_time(time_list, min_value):
    if isinstance(time_list[0], list):  # 如果是2D列表
        normalized_list = [[(value-min_value)/60 +1 if value > 0 else 0 for value in sub_list] for sub_list in time_list]
    else:  # 如果是1D列表
        normalized_list = [(value-min_value)/60 +1 if value > 0 else 0 for value in time_list]
    # clipped_list = [[min(sub_value, const_time_vocab) for sub_value in value] if isinstance(value, list) else min(value, const_time_vocab) for value in normalized_list]
    return normalized_list

def pad_or_truncate(lst, max_len):
    if len(lst) > max_len:
        # If the list is longer than max_len, take the last max_len elements
        return lst[-max_len:]
    else:
        # If the list is shorter than max_len, pad with zeros at the beginning
        return [0]*(max_len - len(lst)) + lst

def pad_or_truncate_2d(lst, max_len, sub_max_len):
    # First pad or truncate each sublist
    lst = [pad_or_truncate(sublist, sub_max_len) for sublist in lst]

    # Then pad or truncate the main list
    if len(lst) > max_len:
        return lst[-max_len:]
    else:
        return [[0]*sub_max_len]*(max_len - len(lst)) + lst

def pad_or_truncate_3d(lst, max_len, sub_max_len, sub_sub_max_len):
    # First pad or truncate each sublist
    lst = [pad_or_truncate_2d(sublist, sub_max_len, sub_sub_max_len) for sublist in lst]

    # Then pad or truncate the main list
    if len(lst) > max_len:
        return lst[-max_len:]
    else:
        return [[[0]*sub_sub_max_len]*sub_max_len]*(max_len - len(lst)) + lst
    


def my_collate_fn(batch):
    # Extract the elements from the batch
    rec_sessions,  query_sessions,  \
                query_sessions_time_normalized,  rec_sessions_time_normalized, session_time_list_normalized, \
                user_id, time, lag_time, label, session_label_list = zip(*batch)
    
    # Pad the sequences to the same length
    rec_sessions = torch.tensor(rec_sessions)
    query_sessions = torch.tensor(query_sessions)
    user_id = torch.tensor(user_id)
    time = torch.tensor(time)
    label = torch.tensor(label)
    query_sessions_time_normalized = torch.tensor(query_sessions_time_normalized).float()
    rec_sessions_time_normalized = torch.tensor(rec_sessions_time_normalized).float()
    session_time_list_normalized = torch.tensor(session_time_list_normalized).float()
    session_label_list = torch.tensor(session_label_list)
    lag_time = torch.tensor(lag_time)




    return  rec_sessions,  query_sessions,  \
            query_sessions_time_normalized,  rec_sessions_time_normalized, session_time_list_normalized, \
            user_id, time,  lag_time, label, session_label_list

