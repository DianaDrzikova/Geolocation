
import torch 
from torch import nn

class FullGeological():
    def __init__(self, model):
        self.model = model

    def load(self):

        self.model.mid.reg = torch.nn.Sequential(
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 127*2)
        )

        self.model.mid.reg_country = torch.nn.Sequential(
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 7 * 2),  # 7 countries × (lat, lon)
        )

        self.model.mid.reg_admin1 = torch.nn.Sequential(
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 58 * 2),  # 58 admin1 × (lat, lon)
        )

        self.model.mid.reg_admin2 = torch.nn.Sequential(
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 127 * 2),  # 127 admin2 × (lat, lon)
        )

        return self.model


class SimpleRegression():
    def __init__(self, model):
        self.model = model

    def load(self):

        self.model.mid.reg = torch.nn.Sequential(
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 2)
        )

        return self.model
    

class ClassificationAdmin():
    def __init__(self, model):
        self.model = model

    def load(self, out):
        self.model.mid.reg = torch.nn.Sequential(
                    torch.nn.LayerNorm(1024),
                    torch.nn.Linear(1024, 512),
                    torch.nn.GELU(),
                    torch.nn.Linear(512, 2)
                )
        self.model.mid.admin = torch.nn.Sequential(
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, out),  
        )
        return self.model

class ClassificationCountry():
    def __init__(self, model):
        self.model = model

    def load(self, out):
        self.model.mid.reg = torch.nn.Sequential(
                    torch.nn.LayerNorm(1024),
                    torch.nn.Linear(1024, 512),
                    torch.nn.GELU(),
                    torch.nn.Linear(512, 2)
                )
        self.model.mid.country = torch.nn.Sequential(
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, out),  
        )
        return self.model

class CLIPRegressorBasic(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        embed_dim = clip_model.visual.output_dim # 768

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6, dtype=torch.float32),
            nn.Linear(embed_dim, 512, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(512, 2, dtype=torch.float32),
            nn.Tanh()
        )

    def forward(self, imgs):
        with torch.no_grad():
            feats = self.clip.encode_image(imgs.half()).float()
        return self.head(feats)
    

class CLIPRegressorBasic(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        embed_dim = clip_model.visual.output_dim # 512

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-6, dtype=torch.float32),
            nn.Linear(embed_dim, 512, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(512, 2, dtype=torch.float32),
            nn.Tanh()
        )

    def forward(self, imgs):
        with torch.no_grad():
            feats = self.clip.encode_image(imgs.half()).float()
        return self.head(feats)
    

class CLIPRegressorGeoClip(nn.Module):
    def __init__(self, gc):
        super().__init__()
        self.gc = gc
        self.image_encoder = gc.image_encoder

        self.head = nn.Sequential(
            nn.LayerNorm(512, eps=1e-6, dtype=torch.float32),
            nn.Linear(512, 512, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(512, 2, dtype=torch.float32),
            nn.Tanh()
        )

    def forward(self, imgs):
        with torch.no_grad():
            feats = self.gc.image_encoder(imgs.half()).float()
        return self.head(feats)