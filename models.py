import layers
import torch.nn as nn 
import torch
import copy
import tqdm
import numpy as np 
import data_utils
from sklearn.preprocessing import StandardScaler


class STGCN(nn.Module):
    def __init__(self,  K, n_vertex, k_t, graph_kernel, n_his=12):
        super(STGCN, self).__init__()
   
        self.stconvblock = layers.STConvBlock(K=K, n_vertex=n_vertex, k_t=k_t, graph_kernel=graph_kernel, 
                                               channels=[1, 16, 64])
        self.stconvblock2 = layers.STConvBlock(K=K, n_vertex=n_vertex, k_t=k_t, graph_kernel=graph_kernel, 
                                               channels=[64, 16, 64])

        self.outputblock = layers.OutputBlock(Ko=n_his-2*2*(k_t-1), n_vertex=n_vertex, channels=[64, 128, 1]) 
        # Ko = kernel size of temporal conv layer in output block
        # this kernel size must be set to n_his-num_blocks*2*(kt-1)  (so that the output time step is 1)
        # every Spatio-temporal Convolutional Blocfk output tensor with dimension = [(M-2(kt-1)), n, cl+1]. so every block reduces M by 2(kt-1)

    def __call__(self, x):
        x = self.stconvblock(x)
        x = self.stconvblock2(x)
        x = self.outputblock(x)
        return x


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
    

def multi_pred(model, n_pred, n_his, x):
    '''
    1 batch multi step prediction.

    return:
        step_list: list of y_pred length n_pred, each element is (bs, n_vertex). step_list dim = [n_pred, bs, n_vertex]
    '''
    step_list = []
    test_seq = x[:, 0:n_his, : , : ] # bs, ts, n_vertex, c_in
    for _ in range(n_pred):
        y_pred = model(test_seq) # [32, 1, 1, 228] or [bs, c_out=1, ts, n_vertex]
        test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
        test_seq[:, n_his - 1, :, :] = y_pred.view(len(x), x.shape[2], 1) # [bs, ts, n_vertex, c_in]
        step_list.append(y_pred.view(len(x), -1).cpu().numpy())
    return step_list # [n_pred, bs, n_vertex]

def train(train_iter, val_iter, model, n_pred, n_his, scaler, early_stopping_patience, learning_rate, metrics_history,
           epochs, save_every_epoch):
    print('training model started! 🔥🔥🔥')
    es = EarlyStopping(patience=early_stopping_patience, min_delta=0, restore_best_weights=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    done = False 
    # All the tests use 60 minutes as the historical time window, a.k.a. 12 observed
    # data points (M = 12) are used to forecast traffic conditions in the next 15, 30, and 45 minutes (H = 3, 6, 9).
    # H steps:
    h_step_idx = np.arange(3, n_pred+1, 3) -  1 # 3,6,9 time step for val & test if n_pred=9

    for epoch in range(epochs):
       
        if done:
            break
        model.train()
        train_loss, train_n = 0.0, 0
        for x, y in tqdm.tqdm(train_iter):
            # y shape [bs, n_pred, n_vertex]
            optimizer.zero_grad()
            y_pred = model(x).view(len(x), -1) # [bs, n_vertex*1*1] = [bs, n_vertex] prediction for next 5 minutes/t+1
            loss = loss_fn(y_pred, y[:, 0, :]) # only use the first time step t+1 for training
            loss.backward()
            optimizer.step()    
            train_loss += loss.item() * y.shape[0]
            train_n += y.shape[0]
            
        scheduler.step()
        metrics_history['train_loss'].append(train_loss/train_n)

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in tqdm.tqdm(val_iter):
                y_pred = model(x).view(len(x), -1) # [bs, n_vertex*1*1] = [bs, n_vertex] prediction for next 5 minutes/t+1
                loss = loss_fn(y_pred, y[:, 0, :]) # only use the first time step t+1 for training
                val_loss += loss.item() * y.shape[0]
                val_n += y.shape[0]

        
        metrics_history['val_loss'].append(val_loss/val_n)

        if es(model, val_loss/val_n):
            print(es.status)
            done = True

        print(f'Epoch {epoch}, Train Loss: {train_loss/train_n}, Validation Loss: {val_loss/val_n}')
      
        if epoch % save_every_epoch == 0:
            torch.save(model.state_dict(), f'stgcn_epoch_{epoch}.pth')
            print(f'Model saved to stgcn_epoch_{epoch}.pth')
    print('Training model finished! 🌊🌊🌊')

    



def evaluation(model, y_pred, y, scaler: StandardScaler ):
    '''
    y dim = [len(h_step_idx), total_samples, n_vertex]
    y_pred dim = [len(h_step_idx), total_samples, n_vertex]
    
    return: 
        res: np array of shape (n_pred*3,) each element is MAPE, RMSE, MAE for each prediction horizon
        mean_mse: mean mse value of each time step prediction
    '''

    dim = len(y_pred.shape)
    if dim == 2:
        # dim = [total_samples, n_vertex]
        y_unnorm = scaler.inverse_transform(y)
        y_pred_unnorm = scaler.inverse_transform(y_pred)
        return np.array([data_utils.mape(y=y_unnorm, y_pred=y_pred_unnorm), data_utils.rmse(y=y_unnorm, y_pred=y_pred_unnorm),
                         data_utils.mae(y=y_unnorm, y_pred=y_pred_unnorm)]), data_utils.mse(y=y_unnorm, y_pred=y_pred_unnorm),

    else:
        # dim = [len(h_step_idx), total_samples, n_vertex]
        tmp_list = [] # (len(h_step_idx), 3)
        tmp_mse_list = []
        # recurse each time step H, to get MAPE, RMSE, MAE, and MSE for each time step
        for i in range(y_pred.shape[0]):
            tmp_res, mse = evaluation(model, y_pred[i], y[i], scaler)
            tmp_list.append(tmp_res)
            tmp_mse_list.append(mse)
        # get min mse 
        mean_mse = float(np.mean(tmp_mse_list))

        return np.concatenate(tmp_list, axis=-1), mean_mse # np.concatenate(tmp_list, axis=-1) = [n_pred*3]



def test(test_iter, model, scaler, n_pred, n_his):
    print('testing model started! 🔥🔥🔥')
    model.eval()
    h_step_idx = np.arange(3, n_pred+1, 3) -  1 # 3,6,9 time step for val & test if n_pred=9
    test_pred_list = []
    all_ys = []

    with torch.no_grad():
        for x, y in tqdm.tqdm(test_iter):
            all_ys.append(y)
            step_list = multi_pred(model, n_pred, n_his, x) # [n_pred, bs, n_vertex]
            test_pred_list.append(step_list) # list of [n_pred, bs, n_vertex], dim = [total_batch, n_pred, bs, n_vertex]

    all_ys = torch.cat(all_ys, dim=0)  # shape: [total_samples, n_pred, n_vertex]
    test_pred_array = np.concatenate(test_pred_list, axis=1)  # [n_pred, total_samples, n_vertex]
    test_pred_H = test_pred_array[h_step_idx] # [len(h_step_idx), total_samples, n_vertex]
    test_samples_size = test_pred_array.shape[1]

    y_ground_truth_H = all_ys[:test_samples_size, h_step_idx, :].permute(1,0,2).cpu().numpy() # [len(h_step_idx), total_samples, n_vertex] 
    test, mean_mse_test = evaluation(model, test_pred_H, y_ground_truth_H, scaler)
    
    for i in range(len(h_step_idx)):
        h_idx =  h_step_idx[i]
        print( f', prediction time step: { (h_idx +1 )  } or next {(h_idx+1)*5} minutes , MAPE {test[i*3]}, RMSE {test[i*3+1]}, MAE {test[i*3+2]}')

    print('Testing model finished! 🌊🌊🌊')
