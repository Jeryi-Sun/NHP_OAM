#coding=utf-8


"""data files info"""

root_path = './data/'



train_file = root_path+'train_dataset.csv'
all_train_file = root_path+'past_and_train.csv'
valid_file = root_path+'valid_dataset.csv'
test_file = root_path+'test_dataset.csv'
recommendation_index = root_path+"rec_sequence.pickle"
query_index = root_path+"query_sequence.pickle"
item_vocab = root_path+'item2id.pickle'
user_vocab = root_path+"user2id.pickle"

"""item/user/query feature"""

item_id_num = 287695 + 1 #zero for padding
item_id_dim = 64
item_type1_num = 38
item_type1_dim = 8
item_cate_num = 37
item_cate_dim = 8


user_id_num = 798086+1
user_id_dim = 64
user_gender_num = 3
user_gender_dim = 4
user_age_num = 8
user_age_dim = 4
user_src_level_num = 4
user_src_level_dim = 4


query_id_num = 127456 + 1
query_id_dim = 64
query_search_source_num = 4
query_search_source_dim = 48

"""experiment config"""
ours_time_dim= 64
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


