#coding=utf-8


"""data files info"""

root_path = './data/' # set the root path of data files


train_file = root_path+'train.csv'
all_train_file = root_path+'all_train.csv' # train_file + past_history_file
valid_file = root_path+'valid.csv'
test_file = root_path+'test.csv'
all_file = root_path+'all_data.csv' # past_history_file + train_file + valid_file + test_file


open_search_index = root_path+"open_search_index.pickle"
search_index = root_path+"src_sequence.pickle"
recommendation_index = root_path+"rec_sequence.pickle"
query_index = root_path+"query_sequence.pickle"
item_vocab = root_path+'item2id.pkl'
user_vocab = root_path+"user2id.pkl"

"""item/user/query feature"""

item_id_num = 2542883 + 1 #zero for padding
item_id_dim = 32
item_type1_num = 38
item_type1_dim = 8
item_cate_num = 37
item_cate_dim = 8


user_id_num = 25877+1
user_id_dim = 32
user_gender_num = 3
user_gender_dim = 4
user_age_num = 8
user_age_dim = 4
user_src_level_num = 4
user_src_level_dim = 4


query_id_num = 107499 + 2 
query_id_dim = 32
query_search_source_num = 4
query_search_source_dim = 48

time_feature_dim=6
"""experiment config"""
max_seq_len_reco = 50
max_seq_len_search = 10
max_seq_len_open_search = 5


# our params
ours_time_dim= 32
ours_session_len = 10
ours_max_query_len = 5
ours_max_word_len = 5
ours_max_search_len = 10
ours_max_reco_len = 20
ours_week_num = 8
ours_hour_num = 25
ours_min_num = 61
ours_second_num = 61
ours_whms_dim = 8
ours_lag_time_vocab = 1440+1


