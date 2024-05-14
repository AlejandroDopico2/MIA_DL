import numpy as np 
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from typing import Tuple 
import plotly.graph_objects as go 
from typing import List, Dict, Union 


class FID:
    def __init__(self, img_size: Tuple[int, int, int] = (299, 299, 3)):
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=img_size)
        self.img_size = img_size 
        
    def __call__(self, imgs1: np.ndarray, imgs2: np.ndarray) -> float:
        # scale images 
        imgs1, imgs2 = map(self.preprocess, (imgs1, imgs2))
        
        act1 = self.model.predict(imgs1, verbose=0)
        act2 = self.model.predict(imgs2, verbose=0)
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid 

    def preprocess(self, imgs: np.ndarray) -> np.ndarray:
        return np.stack([preprocess_input(resize(img, self.img_size, 0)) for img in imgs], 0)
        


def plot_history(history: Dict[str, List[float]], name: Union[str, List[str]] = 'loss') -> go.Figure:
    fig = go.Figure()
    if isinstance(name, str):
        fig.add_trace(go.Scatter(y=history[name], name='train'))
        fig.add_trace(go.Scatter(y=history[f'val_{name}'], name='val'))
    else:
        for x in name:
            fig.add_trace(go.Scatter(y=history[x], name=x))
    fig.update_layout(template='seaborn', height=500, width=800, margin=dict(t=30, b=10, l=30, r=10),
                    title_text=f'Training {name if isinstance(name, str) else ""} evolution')
    fig.update_xaxes(title_text='epochs')
    if isinstance(name, str):
        fig.update_yaxes(title_text=name)
    return fig 
