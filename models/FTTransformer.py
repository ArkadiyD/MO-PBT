import rtdl
import torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(self, dim, cat_cardinalities, dropout, **kwargs):
        super().__init__()

        self.model = rtdl.FTTransformer.make_default(
            n_num_features=dim,
            cat_cardinalities=cat_cardinalities,
            last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=2,
        )
        
        
    def update(self, dropouts):
        for name, module in self.model.named_modules():
            classname = module.__class__.__name__
            if 'Dropout' in classname:
                if 'attention' in name and 'residual' not in name:
                    setattr(module, 'p', dropouts[0])
                elif 'ffn' in name and 'residual' not in name:
                    setattr(module, 'p', dropouts[1])
                elif 'residual' in name:
                    setattr(module, 'p', dropouts[2])

        
    def forward(self, x_num, x_cat=None):
        return self.model(x_num, x_cat=x_cat)#.flatten()
