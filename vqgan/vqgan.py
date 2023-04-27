import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import Namespace
from tqdm.notebook import tqdm
import torchvision.utils as vutils
default_args = Namespace(
    latent_dim=4,
    image_size=256,
    ngpu = 1,
    num_codebook_vectors=16384,
    beta=0.25,
    z_channels = 4,
    image_channels=3,
    dataset_path= "wikiart_images",
    device="cuda",
    batch_size=2,
    epochs=100,
    learning_rate=4.5e-06,
    beta1=0.5,
    beta2=0.9,
    disc_factor=1.0,
    rec_loss_factor=1.0,
    perceptual_loss_factor=1.0,
    channel_multipliers = [1,2,2,4],
    channel = 128,
    num_res_blocks = 2,
    resolution = 256,
    image_key = "image",
    ch = 128,
    in_channels = 3,
    out_ch = 3,
    ch_mult = (1,2,2,4),
    base_learning_rate = 4.5e-06,
    embed_dim = 4,
    n_embed = 16384,
    double_z = False,
    attn_resolutions = (32,),
    dropout = 0.0,
    disc_conditional = False,
    disc_in_channels = 3,
    disc_num_layers = 2,
    disc_start = 1,
    disc_weight = 0.6,
    codebook_weight = 1.0,
    ckpt_path = 'models/vqgan.ckpt',
    ignore_keys = []
)
from vqgan.encoder import Encoder
from vqgan.decoder import Decoder
from vqgan.quantizer import VectorQuantizer
from vqgan.loss import VQLPIPSWithDiscriminator
from vqgan.discriminator import NLayerDiscriminator

class VQModel(pl.LightningModule):
    def __init__(self,
                 args,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.n_embed = args.num_codebook_vectors
        self.embed_dim = args.latent_dim
        self.learning_rate = args.learning_rate * args.ngpu * args.batch_size
        self.z_channels = args.z_channels
        self.image_key = args.image_key
        self.encoder = Encoder(**vars(args)).float()
        self.decoder = Decoder(**vars(args)).float()
        self.loss = VQLPIPSWithDiscriminator(args.disc_start).eval()
        self.quantize = VectorQuantizer(self.n_embed, self.embed_dim, args.beta).float()
        self.quant_conv = torch.nn.Conv2d(self.z_channels, self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.z_channels, 1)
        if args.ckpt_path is not None:
            self.init_from_ckpt(args.ckpt_path, ignore_keys=args.ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
       # print(h.shape)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[0]
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        if batch_idx % 25 == 0:
            self.log_images(x, xrec)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, x, xrec, **kwargs):
        images = torch.cat((x.add(1).mul(0.5), xrec.add(1).mul(0.5)), dim = 0)
       # print("catted shape ", images.shape)
        tb = self.logger.experiment
        grid = vutils.make_grid(images, nrow = x.shape[0], dataformats = "NCHW")
        tb.add_image("Real vs Fake", grid, self.global_step)

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

if __name__ == "__main__":
    from data import load_data
    import argparse
    parser = argparse.ArgumentParser(description = "training VQGAN")
    parser.add_argument("--ckpt_path", default = None, help = "checkpoint path")
    args = parser.parse_args()

    model = VQModel(default_args)
    trainer = pl.Trainer(max_epochs = 10, accelerator = "gpu")
    train_loader = load_data(default_args.dataset_path, default_args.batch_size)
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = None, ckpt_path = args.ckpt_path)
