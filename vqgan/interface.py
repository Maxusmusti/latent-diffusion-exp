import torch.nn as nn
from vqgan import VQModel, default_args

def get_default_args():
    return default_args

class VQInterface(VQModel):    
    def __init__(self, args):
        super().__init__(args)
        self.embed_dim = args.embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, quantize_first=True):
        if quantize_first:
            quant, _, _ = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    

def pretrained_vqgan():
    args = get_default_args()
    ckpt_path = default_args.ckpt_path
    model = VQInterface(args)
    model.init_from_ckpt(ckpt_path)
    return model