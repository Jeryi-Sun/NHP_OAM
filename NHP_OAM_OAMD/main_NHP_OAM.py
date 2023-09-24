import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import const
import random
import os
import setproctitle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score
import numpy as np
import logging
from dataset.dataset_ours_THP import MyDataset, my_collate_fn
from datetime import datetime
os.environ['NUMEXPR_MAX_THREADS'] = '48'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
setup_seed(1)

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, help='experiment name', default='default')
parser.add_argument('--log', type=str, help='exp details, used for log name', default='cat')
parser.add_argument('--workspace', type=str, default='./workspace')
parser.add_argument('--dataset_name', type=str, default='kuaishou')
parser.add_argument('--use_cpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)
parser.add_argument('--gpu_id', type=str, default='1')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, help='training batch_size', default=256)
parser.add_argument('--early_stop', type=int, help='early_stop_num', default=10)
parser.add_argument('--data_roll_num', type=int, help='data_roll_number', default=0)
parser.add_argument('--session_len', type=int, help='session_len', default=10)
parser.add_argument('--alpha', type=float, help='session_len', default=0.0001)


args = parser.parse_args()


from models.NHP_OAM import NHP_OAM, ModelArgs






# Get current date and time
now = datetime.now()

# Format the datetime string to the desired format, here 'YearMonthDayHour'
dt_string = now.strftime("%Y%m%d%H")

# Set up logging
logging.basicConfig(filename=f'./logs/{dt_string}_log_{args.log}.log', level=logging.INFO, 
                    format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()


logger.info("Arguments: %s", args)

setproctitle.setproctitle("求一张卡")

def train(model, train_loader, val_loader, test_loader, lr=0.001, epochs=10, device='cpu', save_path=None):
    """
    Train a model with given train_loader and val_loader. Evaluation on test_loader is added.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The DataLoader for training data.
        val_loader (DataLoader): The DataLoader for validation data.
        test_loader (DataLoader): The DataLoader for testing data.
        lr (float, optional): Learning rate. Defaults to 0.001.
        epochs (int, optional): Number of epochs. Defaults to 10.
        device (str, optional): Device to use for training. Defaults to 'cpu'.
        save_path (str, optional): Path to save the best model parameters. Defaults to None.

    Returns:
        model (nn.Module): The trained model.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    best_acc = 0.0
    best_f_0_5_valid = 0.0
    best_valid_epoch = 0
    best_f_0_5_test = 0.0
    best_test_epoch = 0
    best_number = 0

    best_threshold = 0.5  # Initial threshold
    all_loss = []
    for epoch in tqdm(range(epochs)):
        model.train()
        for rec_sessions, search_sessions, query_sessions,  \
            query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, \
            user_id, time, lag_time, label, session_label_list in tqdm(train_loader):

            rec_sessions = rec_sessions.to(device)
            search_sessions = search_sessions.to(device)
            query_sessions = query_sessions.to(device)
            query_sessions_time_normalized = query_sessions_time_normalized.to(device)
            search_sessions_time_normalized = search_sessions_time_normalized.to(device)
            rec_sessions_time_normalized = rec_sessions_time_normalized.to(device)
            session_time_list_normalized = session_time_list_normalized.to(device)
            user_id = user_id.to(device)
            time = time.to(device)
            label = label.to(device)
            session_label_list = session_label_list.to(device)
            lag_time = lag_time.to(device)

            loss = model.train_(rec_sessions, search_sessions, query_sessions,  
                                query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, 
                                user_id, time, lag_time, label, session_label_list)
            
            loss.backward()
            all_loss.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        logger.info(f'Epoch {epoch+1}/{epochs}: train loss {np.mean(all_loss):.4f}')
        print(f'Epoch {epoch+1}/{epochs}: train loss {np.mean(all_loss):.4f}')

        model.eval()
        y_true, y_pred = [], []
        y_scores = []  # store predicted probabilities for ROC AUC computation
        with torch.no_grad():
            for rec_sessions, search_sessions, query_sessions,  \
                query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, \
                user_id, time, lag_time, label, session_label_list in tqdm(val_loader):
                rec_sessions = rec_sessions.to(device)
                search_sessions = search_sessions.to(device)
                query_sessions = query_sessions.to(device)
                query_sessions_time_normalized = query_sessions_time_normalized.to(device)
                search_sessions_time_normalized = search_sessions_time_normalized.to(device)
                rec_sessions_time_normalized = rec_sessions_time_normalized.to(device)
                session_time_list_normalized = session_time_list_normalized.to(device)
                user_id = user_id.to(device)
                time = time.to(device)
                label = label.to(device)
                session_label_list = session_label_list.to(device)
                lag_time = lag_time.to(device)


                output = model.infer_(rec_sessions, search_sessions, query_sessions,  
                                      query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, 
                                      user_id, time, lag_time, session_label_list)
                    
                y_true += label.cpu().numpy().tolist()
                y_scores += output.cpu().numpy().tolist()


            # Determine the best threshold based on F0.5 score
            thresholds = list(np.arange(0, 1, 0.01))
            f_0_5_scores = []
            for threshold in thresholds:
                output_class = (np.array(y_scores) > threshold).astype(float)  # threshold output to obtain class predictions
                f_0_5_score = fbeta_score(y_true, output_class, beta=0.5)
                f_0_5_scores.append(f_0_5_score)
            
            best_idx = np.argmax(f_0_5_scores)
            best_threshold = thresholds[best_idx]

            output_class = (np.array(y_scores) > best_threshold).astype(float) 
            y_pred += output_class.tolist()

            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            f_0_5 = fbeta_score(y_true, y_pred, beta=0.5)
            auc = roc_auc_score(y_true, y_scores)  # compute ROC AUC
            
            logger.info(f'Epoch {epoch+1}/{epochs}: val accuracy {acc:.4f}, precision {precision:.4f}, recall {recall:.4f}, F1-score {f1:.4f}, F0.5-score {f_0_5:.4f}, AUC {auc:.4f}, Best Threshold {best_threshold:.2f}')
            print(f'Epoch {epoch+1}/{epochs}: val accuracy {acc:.4f}, precision {precision:.4f}, recall {recall:.4f}, F1-score {f1:.4f}, F0.5-score {f_0_5:.4f}, AUC {auc:.4f}, Best Threshold {best_threshold:.2f}')

            if f_0_5>best_f_0_5_valid:
                best_f_0_5_valid=f_0_5
                score_list = [acc, precision, recall, f1, f_0_5, auc]
                best_valid_epoch = epoch+1
                best_number = 0
            else:
                best_number += 1

            if best_number >= args.early_stop:
                break

            if save_path is not None:
                torch.save(model.state_dict(), save_path)



        # Test phase
        y_true_test, y_pred_test, y_scores_test = [], [], []
        with torch.no_grad():
            for rec_sessions, search_sessions, query_sessions,  \
                query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, \
                user_id, time, lag_time, label, session_label_list in tqdm(test_loader):
                rec_sessions = rec_sessions.to(device)
                search_sessions = search_sessions.to(device)
                query_sessions = query_sessions.to(device)
                query_sessions_time_normalized = query_sessions_time_normalized.to(device)
                search_sessions_time_normalized = search_sessions_time_normalized.to(device)
                rec_sessions_time_normalized = rec_sessions_time_normalized.to(device)
                session_time_list_normalized = session_time_list_normalized.to(device)
                user_id = user_id.to(device)
                time = time.to(device)
                label = label.to(device)
                session_label_list = session_label_list.to(device)
                lag_time = lag_time.to(device)


                output = model.infer_(rec_sessions, search_sessions, query_sessions,  
                                      query_sessions_time_normalized, search_sessions_time_normalized, rec_sessions_time_normalized, session_time_list_normalized, 
                                      user_id, time, lag_time, session_label_list)
                
                y_scores_test += output.cpu().numpy().tolist()
                output_class = (output > best_threshold).float()  # threshold output to obtain class predictions
                y_true_test += label.cpu().numpy().tolist()
                y_pred_test += output_class.cpu().numpy().tolist()

            acc_test = accuracy_score(y_true_test, y_pred_test)
            precision_test = precision_score(y_true_test, y_pred_test)
            recall_test = recall_score(y_true_test, y_pred_test)
            f1_test = f1_score(y_true_test, y_pred_test)
            f_0_5_test = fbeta_score(y_true_test, y_pred_test, beta=0.5)
            auc_test = roc_auc_score(y_true_test,  y_scores_test)  # compute ROC AUC

            if f_0_5_test>best_f_0_5_test:
                best_f_0_5_test=f_0_5_test
                score_list_test = [acc_test, precision_test, recall_test, f1_test, f_0_5_test, auc_test]
                best_test_epoch = epoch+1

            logger.info(f'Test phase results - accuracy {acc_test:.4f}, precision {precision_test:.4f}, recall {recall_test:.4f}, F1-score {f1_test:.4f}, F0.5-score {f_0_5_test:.4f}, AUC-score {auc_test:.4f}')
            print(f'Test phase results - accuracy {acc_test:.4f}, precision {precision_test:.4f}, recall {recall_test:.4f}, F1-score {f1_test:.4f}, F0.5-score {f_0_5_test:.4f}, AUC-score {auc_test:.4f}')
    logger.info(f'best valid epoch: {best_valid_epoch}')
    logger.info(f'best valid phase results - accuracy {score_list[0]:.4f}, precision {score_list[1]:.4f}, recall {score_list[2]:.4f}, F1-score {score_list[3]:.4f}, F0.5-score {score_list[4]:.4f}, AUC-score {score_list[5]:.4f}') 
    logger.info(f'best test epoch: {best_test_epoch}')
    logger.info(f'best test phase results - accuracy {score_list_test[0]:.4f}, precision {score_list_test[1]:.4f}, recall {score_list_test[2]:.4f}, F1-score {score_list_test[3]:.4f}, F0.5-score {score_list_test[4]:.4f}, AUC-score {score_list_test[5]:.4f}') 
    return model



if __name__ == '__main__':
    # 加载数据
    if args.data_roll_num == 0:
        train_dataset = MyDataset(const.train_file, const.all_file, session_len=args.session_len)
        val_dataset = MyDataset(const.valid_file,const.all_file, session_len=args.session_len)
        test_dataset = MyDataset(const.test_file, const.all_file, session_len=args.session_len)
    elif args.data_roll_num == 1:
        train_dataset = MyDataset(const.train_file_1, const.all_file, session_len=args.session_len)
        val_dataset = MyDataset(const.valid_file_1,const.all_file, session_len=args.session_len)
        test_dataset = MyDataset(const.test_file_1, const.all_file, session_len=args.session_len)
    elif args.data_roll_num == 2:
        train_dataset = MyDataset(const.train_file_2, const.all_file, session_len=args.session_len)
        val_dataset = MyDataset(const.valid_file_2,const.all_file, session_len=args.session_len)
        test_dataset = MyDataset(const.test_file_2, const.all_file, session_len=args.session_len)
    elif args.data_roll_num == 3:
        train_dataset = MyDataset(const.train_file_3, const.all_file, session_len=args.session_len)
        val_dataset = MyDataset(const.valid_file_3,const.all_file, session_len=args.session_len)
        test_dataset = MyDataset(const.test_file_3, const.all_file, session_len=args.session_len)
    else:
        print('dataset number error')
        exit(-1)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=my_collate_fn, pin_memory = True, prefetch_factor = args.batch_size // 16 * 2, num_workers = 16)
    val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=my_collate_fn, pin_memory = True, prefetch_factor = args.batch_size // 16 * 2, num_workers = 16)
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=my_collate_fn, pin_memory = True, prefetch_factor = args.batch_size // 16 * 2, num_workers = 16)


    """
    增加数据类
    """
    device = 'cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu'

    model_args_reco: ModelArgs = ModelArgs(
    Dense_dim=128,
    item_dim=const.item_id_dim,
    item_num=const.item_id_num,
    word_num=const.query_id_num,
    time_num=const.ours_lag_time_vocab+1,
    time_dim=const.ours_time_dim,
    user_num = const.user_id_num,
    user_dim = const.user_id_dim,
    device = device
    )


    # 定义模型
    model = NHP_OAM(model_args_reco,alpha=args.alpha)

    # 训练模型
    model.to(device)

    train(model, train_loader, val_loader, test_loader, lr=args.lr, epochs=args.epochs, device=device)