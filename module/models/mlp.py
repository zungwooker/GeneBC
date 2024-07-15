import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self, in_features, out_features, num_layers, hidden_dim: list[int]) -> None:
#         super().__init__()
#         assert num_layers == len(hidden_dim), "Check num_layers and hidden_dim."
            
#         layers = [nn.Flatten()]
#         d_prev = in_features
#         for dim in hidden_dim:
#             layers += [nn.Linear(d_prev, dim), nn.ReLU()]
#             d_prev = dim
        
#         self.layers = nn.Sequential(*layers)
#         self.classifier = nn.Linear(d_prev, out_features)
        
#     def forward(self, x):
#         x = self.layers(x)
#         return self.classifier(x)

class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        x = self.feature(x)
        final_x = self.classifier(x)

        if return_feat:
            return final_x, x
        else:
            return final_x