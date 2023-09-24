import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from .transformer.Models import Transformer
import math
import models.utils as utils
@dataclass
class ModelArgs:
    '''
    model params
    '''
    Dense_dim: int
    item_dim: int
    time_dim: int
    item_num: int
    word_num: int
    d_model: int = 32
    time_num: int = 1000
    user_num: int = 1000
    user_dim: int = 32
    n_head: int = 1 
    device: str = 'cuda:1'


class query_and_item_feat(nn.Module):
    def __init__(self, Model_param:ModelArgs):
        super().__init__()
        self.item_embedding = nn.Embedding(num_embeddings=Model_param.item_num, embedding_dim=Model_param.item_dim)
        self.word_emb = nn.Embedding(num_embeddings=Model_param.word_num, embedding_dim=Model_param.item_dim)
        self.time_emb = nn.Embedding(num_embeddings=Model_param.time_num, embedding_dim=Model_param.item_dim)
        self.user_emb = nn.Embedding(num_embeddings=Model_param.user_num, embedding_dim=Model_param.user_dim)

    def get_reco_emb(self, x):

        return self.item_embedding(x), self.make_mask(x)
    
    def get_search_emb(self, x):

        return self.item_embedding(x), self.make_mask(x)
    
    def get_word_emb(self, x):
        word_embedding = self.word_emb(x)
        word_mask = 1-self.make_mask(x)
        mask_sum = torch.sum(word_mask, dim=3, keepdim=True)  # 在query_len维度上求和
        mask_sum[mask_sum == 0] = 1  # 防止除以0的情况，如果一个query全是mask的话，那么在这个query上的mask_sum就是0，将其设为1
        word_embedding_sum = torch.sum(word_embedding, dim=3)  # 在query_len维度上求和
        sentence_embedding = word_embedding_sum / mask_sum  # 取平均
        sentence_mask = torch.eq(torch.sum(word_mask, dim=3), 0).float()  # 在query_len维度上求和
        sentence_mask = sentence_mask.to(x.device)
        
        return sentence_embedding, sentence_mask
    
    def get_time_emb(self, x):
        return self.time_emb(x), self.make_mask(x)
    
    def get_user_emb(self, x):
        return self.user_emb(x)


    def make_mask(self, x):
        mask = torch.eq(x, 0).float()   # 如果 x 等于 0，则 mask 对应位置为 1，否则为 0
        mask = mask.to(x.device)  # 确保 mask 在与 x 相同的设备上
        return mask
    



class NHP_OAM(nn.Module):
    def __init__(self, Model_param:ModelArgs,alpha=0.001):
        super(NHP_OAM, self).__init__()

        Dense_dim = Model_param.Dense_dim
        item_dim = Model_param.item_dim
        time_dim = Model_param.time_dim
        self.args = Model_param

        self.item_query_emb = query_and_item_feat(Model_param)

        self.user_rep = nn.Sequential(
           nn.Linear(item_dim+Model_param.user_dim, Dense_dim),
           nn.ReLU(),
           nn.Linear(Dense_dim, Dense_dim//2),
           nn.ReLU(),
           nn.Linear(Dense_dim//2, 1),
        )

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / item_dim) for i in range(item_dim)],
            device=Model_param.device)

        self.time_linear_short = nn.Linear(item_dim,item_dim)
        self.time_linear_long = nn.Linear(item_dim,item_dim)
        self.time_linear = nn.Linear(time_dim,item_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=item_dim, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        self.THP = Transformer(num_types=2, d_model=item_dim, d_rnn=item_dim, d_inner=1024, n_head=Model_param.n_head, d_k=int(item_dim/Model_param.n_head), d_v=int(item_dim/Model_param.n_head), n_layers=2, device = Model_param.device)

        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss(reduction='none')
        self.alpha = alpha

    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result 

    def forward(self, rec_sessions, search_sessions, query_sessions,  
                query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, 
                user_id, time, lag_time, session_label_list):
        rec_emb, rec_emb_mask = self.item_query_emb.get_reco_emb(rec_sessions)
        search_emb, search_emb_mask = self.item_query_emb.get_search_emb(search_sessions)
        query_emb, query_emb_mask = self.item_query_emb.get_word_emb(query_sessions)

        # rec_emb_mask 的形状为 batch_size * session_len * rec_len, 请对最后一个纬度求和，得到形状为 batch_size * session_len 的 A张量
        # query_emb_mask 的形状为 batch_size * session_len * query_len, 请对最后一个纬度求和，得到形状为 batch_size * session_len B的张量
        # B 除（A+B）计算得到 query_ratio 形状为 batch_size * session_len 

        query_ratio = torch.sum(1-query_emb_mask, dim=-1) / (torch.sum(1-query_emb_mask, dim=-1) + torch.sum(1-rec_emb_mask, dim=-1)+1e-8)




        # 将 rec_emb, search_emb, query_emb 与他们对应的时间戳一起 concat 到一个新的维度
        rec = torch.cat([rec_emb, rec_sessions_time_normalized.unsqueeze(3)], dim=-1)
        search = torch.cat([search_emb, search_sessions_time_normalized.unsqueeze(3)], dim=-1)
        query = torch.cat([query_emb, query_sessions_time_normalized.unsqueeze(3)], dim=-1)

        # 将 rec, search, query concat 到一起
        his_rep = torch.cat([rec, search, query], dim=2)

        # 对 his_rep 的最后一个维度进行排序，获取排序后的索引
        _, indices = his_rep[:,:,:,-1].sort(dim=2)

        # 创建一个用于索引的张量
        idx1 = torch.arange(his_rep.size(0)).unsqueeze(1).unsqueeze(2)  # for batch_size
        idx2 = torch.arange(his_rep.size(1)).unsqueeze(0).unsqueeze(2)  # for session_len

        # 使用获取的索引对 his_rep 进行排序
        his_rep = his_rep[idx1, idx2, indices]
        # 提取 his_rep_emb 和 his_rep_time
        his_rep_emb = his_rep[:,:,:,:-1]

        his_rep_time = his_rep[:,:,:,-1].to(torch.long)



        # 创建 mask
        his_mask = his_rep_time == 0
        his_mask[:,:,-1] = False
        his_mask = his_mask.to(torch.float)

        # 创建 time embedding
        time_embedding = self.temporal_enc(his_rep_time)

        his_rep_time_emb = his_rep_emb + time_embedding
        


        # 将 his_rep_emb 和 his_mask 传入 transformer
        session_level_embedding_list = []
        for i in range(his_rep.size(1)):
            output = self.transformer(his_rep_time_emb[:,i,:,:], src_key_padding_mask=his_mask[:,i,:])
            session_level_embedding_list.append(torch.mean(output, dim=1))

        session_level_embedding = torch.stack(session_level_embedding_list, dim=1)
        """
        开始上！THP
        """
        enc_output, prediction = self.THP(session_label_list, session_time_list_normalized, session_level_embedding)

        long_interest = enc_output[:,-1,:]
        time_emb, _ = self.item_query_emb.get_time_emb(lag_time.to(torch.long))


        time_gate = self.sigmoid(self.time_linear_short(session_level_embedding_list[-1])+self.time_linear_long(long_interest)+self.time_linear(time_emb))
        result_emb = time_gate*long_interest+(1-time_gate)*session_level_embedding_list[-1]

        # user_emb
        user_emb = self.item_query_emb.get_user_emb(user_id)

        # 传入 self.user_rep 得到预测结果
        pred = self.user_rep(torch.cat([result_emb, user_emb], dim=-1))

        return enc_output, prediction, pred, query_ratio

    def train_(self, rec_sessions, search_sessions, query_sessions,  
            query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, 
            user_id, time, lag_time, label, session_label_list):
        
        enc_out, prediction, pred, query_ratio = self.forward(rec_sessions, search_sessions, query_sessions,  
            query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, 
            user_id, time, lag_time, session_label_list)
        
        # 对query_ratio shape batch_size * session_len 在最后一个纬度上进行normalize到 0-1
        min_val = query_ratio.min(dim=-1, keepdim=True)[0]
        max_val = query_ratio.max(dim=-1, keepdim=True)[0]

        query_ratio = (query_ratio - min_val) / (max_val - min_val + 1e-8)
        
        # negative log-likelihood
        event_ll, non_event_ll = utils.log_likelihood(self, enc_out, session_time_list_normalized, session_label_list, query_ratio=query_ratio)
        # event_loss = -torch.mean(event_ll)
        event_loss = -torch.mean(event_ll - non_event_ll)
        

        # type prediction
        pred_loss = self.loss_func(self.sigmoid(pred+query_ratio[:, -1].unsqueeze(1)), label.unsqueeze(1).to(torch.float)).mean()

        # time prediction
        se = utils.time_loss(prediction[1], session_time_list_normalized)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss =  self.alpha * event_loss + pred_loss #+ se / scale_time_loss
        #loss = pred_loss
        
        return loss

    @torch.inference_mode()
    def infer_(self, rec_sessions, search_sessions, query_sessions,  
            query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, 
            user_id, time, lag_time, session_label_list):
        
        _, _, pred, query_ratio = self.forward(rec_sessions, search_sessions, query_sessions,  
            query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, 
            user_id, time, lag_time, session_label_list)
        output = self.sigmoid(pred+query_ratio[:, -1].unsqueeze(1))
        
        return output