import torch.nn as nn

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
class MeanPooling(nn.Module):
    def __init__(self,input_dim=1024,n_classes=1,dropout=True):
        super(MeanPooling, self).__init__()

        
        #self.dp = nn.Dropout(0.25) if dropout else nn.Identity()
        self.dp = nn.Identity()
        self.pool = nn.AdaptiveAvgPool1d(1)
        #self.pool = Attention(input_dim)
        self.head = nn.Linear(input_dim, n_classes)

        self.apply(initialize_weights)

    def forward(self,x):
        # b,p,1024
        #return self.head(self.pool(self.dp(x)))
        return self.head(self.pool(self.dp(x).transpose(-1,-2)).squeeze(-1))

class MaxPooling(nn.Module):
    def __init__(self,input_dim=1024,n_classes=2):
        super(MaxPooling, self).__init__()

        self.head=nn.Linear(input_dim,n_classes)

    def forward(self,x):
        x,_ = self.head(x).max(axis=1)
        return x