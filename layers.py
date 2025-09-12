
import torch.nn as  nn
import numpy as np 
import torch 

class STConvBlock(nn.Module):
    def __init__(self, K, n_vertex, k_t, graph_kernel, channels,):
        super(STConvBlock, self).__init__()
        c_in, c_st, c_o = channels
        self.temporal_gated_conv = TemporalConvLayer( k_t=k_t, c_in=c_in, c_out=c_st, n_vertex=n_vertex)
        self.graph_conv = GraphConv( K=K, c_in=c_st, c_out=c_st, graph_kernel=graph_kernel)
        self.temporal_gated_conv2 = TemporalConvLayer( k_t=k_t, c_in=c_st, c_out=c_o, n_vertex=n_vertex)
        self.layer_norm = nn.LayerNorm(eps=1e-12, normalized_shape=[n_vertex, c_o])
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # bs, ts, n_vertex, c_in
        # equation (8)
        x = self.temporal_gated_conv(x)  # (bs, ts-k_t+1, n_vertex, c_out)
        x = self.graph_conv(x)           
        x = self.temporal_gated_conv2(x)  # (bs, ts-k_t+1, n_vertex, c_out)
        x = self.layer_norm(x)  # (bs, ts-k_t+1, n_vertex, c_out)
        x = self.dropout(x)
        
        return x
    
class GraphConv(nn.Module):
    def __init__(self, K, c_in, c_out, graph_kernel ):
        super(GraphConv, self).__init__()
        self.K = K
        self.c_in = c_in
        self.c_out = c_out
        self.graph_kernel = graph_kernel # (n_vertex, n_vertex*K)
        self.theta = nn.Parameter(torch.randn(K * c_in, c_out)) # k*c_in, c_out
        self.relu = nn.ReLU()
        self.align = Align(c_in=c_in, c_out=c_out)

    def forward(self, x):
        # x =  (bs, ts, n_vertex, c_in)
        x_in = self.align(x)
        n = x.shape[2]
        ts = x.shape[1]

        x_two = x.reshape(-1, n, self.c_in) #  (bs*ts), n_vertex, c_in
        x_tmp = x_two.permute(0, 2, 1).reshape(-1, n)  # (bs*ts)*c_in, n_vertex
        x_mul = torch.matmul(x_tmp, self.graph_kernel).reshape(-1, self.c_in, self.K, n) # (bs*ts), c_in, K, n_vertex
        x_ker = x_mul.permute(0, 3, 1, 2).reshape(-1, self.c_in * self.K)  # (bs*ts)*n_vertex, c_in*K
        x_gconv = torch.matmul(x_ker, self.theta).reshape(-1, n, self.c_out) # (bs*ts), n_vertex, c_out        
        x_gconv2 = x_gconv.reshape(-1, ts, n, self.c_out)   # bs, ts, n_vertex, c_out 
        x_gconv2_out = self.relu(x_gconv2 + x_in) 
        return x_gconv2_out


class OutputBlock(nn.Module):
    def __init__(self, Ko, n_vertex, channels):
        super(OutputBlock, self).__init__()
        c_in, c_out, _ = channels
        self.Ko = Ko
        self.n_vertex = n_vertex
        self.temporal_gated_conv = TemporalConvLayer( k_t=Ko, c_in=c_in, c_out=c_out, n_vertex=n_vertex )
        self.temporal_gated_conv2 = TemporalConvLayer(k_t=1, c_in=c_out, c_out=c_out, n_vertex=n_vertex, act_fn="sigmoid")
        self.fully_connnected = nn.Linear(128, 1, bias=True)

        self.layer_norm = nn.LayerNorm([n_vertex, c_out], eps=1e-12)  
        self.dropout = nn.Dropout(0.3) 
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # bs, ts, n_vertex, c_in
        x = self.temporal_gated_conv(x)  # (bs, ts-k_t+1, n_vertex, c_out)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.temporal_gated_conv2(x) # (bs, ts-1+1, n_vertex, c_out)
        x = self.fully_connnected(x) # bs, ts, n_vertex, c_out=1
        x = x.permute(0, 3, 1, 2) # bs, c_out=1, ts, n_vertex
        return x 


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.conv = nn.Conv2d(in_channels=self.c_in, out_channels=self.c_out, kernel_size=(1, 1), padding='same')
    
    def forward(self, x):
        # x = (bs, ts, n_vertex, c_in)
        if self.c_in > self.c_out:
            # torch Conv2d expects input shape (batch_size, channels, height, width)
            x = x.permute(0, 3, 1, 2) # (bs, c_in, ts, n_vertex)
            x = self.conv(x)
            x = x.permute(0, 2, 3, 1) # (bs, ts, n_vertex, c_out)
        elif self.c_in < self.c_out:
            batch_size, timestep,  n_vertex, _ = x.shape 
            x = torch.cat([x, torch.zeros((batch_size, timestep, n_vertex,  self.c_out - self.c_in), device=x.device)], dim=3)
        return x


class TemporalConvLayer(nn.Module):
    def __init__(self, k_t, c_in, c_out, n_vertex, act_fn="GLU"):
        super(TemporalConvLayer, self).__init__()
        self.k_t = k_t  
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out * 2, kernel_size=(k_t,1))
        self.conv2 = nn.Conv2d(in_channels=c_in, out_channels=c_out , kernel_size=(k_t,1))
        self.align = Align( c_in=c_in, c_out=c_out)
        self.sigmoid = nn.Sigmoid()
        self.act_fn = act_fn

    def forward(self, x):
        # x = (bs, ts, n_vertex, c_in)
        x_in = self.align(x)[: , self.k_t - 1: , : , : ]   

        x = x.permute(0, 3, 1, 2)  # (bs, c_in, ts, n_vertex)
        
        if self.act_fn == "GLU":
            x_conv = self.conv(x) # (bs, c_out*2, ts-k_t+1, n_vertex)
            x_conv = x_conv.permute(0, 2, 3, 1)  # (bs, ts-k_t+1, n_vertex, c_out*2)

            x_p = x_conv[: , : , : , :self.c_out]
            x_q = x_conv[: , :, : , -self.c_out:]
            # equation (7)
            x = torch.mul((x_p+x_in), self.sigmoid(x_q)) # (x_p + residual) ⊙ sigmoid(x_q) . ⊙ =  hadamard/elemet-wise product
            
            return x # (bs, ts-k_t+1, n_vertex, c_out)
        else:
            x_conv = self.conv2(x) # (bs, c_out, ts-k_t+1, n_vertex)
            x_conv = x_conv.permute(0, 2, 3, 1)  # (bs, ts-k_t+1, n_vertex, c_out)
            return self.sigmoid(x_conv)
