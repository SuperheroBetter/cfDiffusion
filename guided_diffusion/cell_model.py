import torch
import torch.nn as nn
import numpy as np

from .nn import (
    linear,
    timestep_embedding,
)

class TimeEmbedding(nn.Module):  
    def __init__(self, hidden_dim):  
        super(TimeEmbedding, self).__init__()  
        self.time_embed = nn.Sequential(  
            nn.Linear(hidden_dim, hidden_dim),  
            nn.SiLU(),  
            nn.Linear(hidden_dim, hidden_dim),  
        )  
        self.hidden_dim = hidden_dim
  
    def forward(self, t):  
        return self.time_embed(timestep_embedding(t, self.hidden_dim).squeeze(1))  

class LabelEmbedding(nn.Module):  
    def __init__(self, input_dim, hidden_dim):  
        super(LabelEmbedding, self).__init__()  
        self.label_embed = nn.Sequential(  
            nn.Linear(input_dim, hidden_dim),  
            nn.SiLU(),  
            nn.Linear(hidden_dim, hidden_dim),  
        )
  
    def forward(self, label):
        return self.label_embed(label)
    
class ResidualBlock(nn.Module):  
    def __init__(self, in_features, out_features, time_features, label_features):  
        super(ResidualBlock, self).__init__()  
        self.fc = nn.Linear(in_features, out_features)  
        self.norm = nn.LayerNorm(out_features) 
        self.emb_time_layer = nn.Sequential(
            nn.SiLU(),
            linear(
                time_features,
                out_features,
            ),
        )
        self.emb_label_layer = nn.Sequential(
            nn.SiLU(),
            linear(
                label_features,
                out_features,
            ),
        )
        self.act = nn.SiLU()  
        self.drop = nn.Dropout(0)  
  
    def forward(self, x, emb_time, emb_label):  
        h = self.fc(x)  
        h = h*self.emb_label_layer(emb_label) + self.emb_time_layer(emb_time)
        h = self.norm(h)
        h = self.act(h)  
        h = self.drop(h)  
        return h
  
class Cell_Unet(nn.Module):  
    def __init__(self, input_dim=2, hidden_dim=[2000,1000,500,500], \
        dropout=0.1, num_classes=12, num_steps=1000, \
        branch = 0, cache_interval=None, non_uniform=False):  
        super(Cell_Unet, self).__init__()  
        self.hidden_dim = hidden_dim  
        
        self.num_classes = num_classes
  
        self.time_embedding = TimeEmbedding(hidden_dim[0])
        
        
        self.label_embedding = LabelEmbedding(sum(num_classes), hidden_dim[0])
  
        # Create layers dynamically  
        self.layers = nn.ModuleList()

        self.layers.append(ResidualBlock(input_dim, hidden_dim[0], hidden_dim[0], hidden_dim[0]))

        for i in range(len(hidden_dim)-1):  
            self.layers.append(ResidualBlock(hidden_dim[i], hidden_dim[i+1], hidden_dim[0], hidden_dim[0]))  
  
        self.reverse_layers = nn.ModuleList()  
        for i in reversed(range(len(hidden_dim)-1)):  
            self.reverse_layers.append(ResidualBlock(hidden_dim[i+1], hidden_dim[i], hidden_dim[0], hidden_dim[0]))  
  
        self.out1 = nn.Linear(hidden_dim[0], int(hidden_dim[1]*2))  
        self.norm_out = nn.LayerNorm(int(hidden_dim[1]*2))
        self.out2 = nn.Linear(int(hidden_dim[1]*2), input_dim, bias=True)

        self.act = nn.SiLU()  
        self.drop = nn.Dropout(dropout)
        
        if non_uniform:
            self.interval_seq, _ = sample_from_quad_center(num_steps, num_steps//cache_interval, 120, 1.5)
        else:
            self.interval_seq = list(range(0, num_steps, cache_interval))
        A = [0]*num_steps
        for i in range(len(self.interval_seq)):
            A[self.interval_seq[i]] = 1
        self.interval_seq = A 
        self.prv_f = None
        self.branch = branch
        self.context_mask = None
  
    def forward(self, x, t, y, inference=False):  
        # convert context to one hot embedding
        if len(self.num_classes) == 1:
            y = nn.functional.one_hot(y, num_classes=self.num_classes[0]).type(torch.float).to(t.device) #[256, 10]
        else:   
            ls_cond = []
            for i in range(y.shape[1]):
                ls_cond.append(nn.functional.one_hot(y[:, i], num_classes=self.num_classes[i]).type(torch.float).to(t.device))
            y = torch.concat(ls_cond, dim=1).type(torch.float).to(t.device)
        
        if inference:
            # print(y.shape, '\n', y)
            y = y.repeat(2, 1)
            # try:
            #     y = y.repeat(2, 1)
            # except:
            #     y = y.repeat(2)
            if self.context_mask == None:
                context_mask = torch.zeros_like(y).to(t.device)
                context_mask[y.shape[0]:] = 1
                self.context_mask = context_mask
            else:
                context_mask = self.context_mask
        else:
            context_mask = torch.bernoulli(torch.zeros_like(y)+0.1).to(t.device)
        
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        y = y * context_mask #[256, 10]
        
        time_emb = self.time_embedding(t)
        label_emb = self.label_embedding(y)
        x = x.float()  
        
        if inference:
            n_layer = len(self.reverse_layers)
            assert 0 <= self.branch < n_layer
            
            if self.interval_seq[t[0]] == 1:
                self.prv_f = None
                
            # Forward pass with history saving  
            history = []
            for i, layer in enumerate(self.layers):
                x = layer(x, time_emb, label_emb)  
                history.append(x)
                if i == self.branch and self.prv_f is not None:
                    break
                    
            if self.prv_f == None:
                if self.branch == n_layer - 1:
                    self.prv_f = history[-1]
                history.pop()
    
                # Reverse pass with skip connections  
                for i, layer in enumerate(self.reverse_layers):  
                    x = layer(x, time_emb, label_emb)  
                    x = x + history.pop()  # Skip connection
                    if self.branch == n_layer - i - 2:
                        self.prv_f = x

                x = self.out1(x)  
                x = self.norm_out(x)
                x = self.act(x)  
                x = self.out2(x)
                
            else:
                # Reverse pass with skip connections
                x = self.prv_f  
                for layer in self.reverse_layers[n_layer-1-self.branch:]:
                    
                    x = layer(x, time_emb, label_emb)  
                    x = x + history.pop()  # Skip connection  
        
                x = self.out1(x)  
                x = self.norm_out(x)
                x = self.act(x)  
                x = self.out2(x)
            
        else:
            # Forward pass with history saving  
            history = []  
            for layer in self.layers:  
                x = layer(x, time_emb, label_emb)  
                history.append(x)  
            
            history.pop()
    
            # Reverse pass with skip connections  
            for layer in self.reverse_layers:  
                x = layer(x, time_emb, label_emb)  
                x = x + history.pop()  # Skip connection  
    
            x = self.out1(x)
            x = self.norm_out(x)
            x = self.act(x)  
            x = self.out2(x)
            
        return x  

def sample_from_quad_center(total_numbers, n_samples, center, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace((-center)**(1/pow), (total_numbers-center)**(1/pow), n_samples+1)
        #print(x_values)
        #print([x for x in np.unique(np.int32(x_values**pow))[:-1]])
        # Raise these values to the power of 1.5 to get a non-linear distribution
        indices = [0] + [x+center for x in np.unique(np.int32(x_values**pow))[1:-1]]
        if len(indices) == n_samples:
            break
        
        pow -=0.02
    return indices, pow
