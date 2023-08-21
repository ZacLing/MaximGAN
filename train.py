import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from maxim_pytorch.maxim_torch import MAXIM_dns_3s
from downsampler import bicubic_downsample

torch.backends.cudnn.benchmark = True

def train_stage(
    stage_id, gen, loader, opt_gen, l1_loss, bce, g_scaler,
):
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    d_scaler = torch.cuda.amp.GradScaler()
    loop = tqdm(loader, leave=True)

    for name, param in gen.named_parameters():
        if name.startswith(f"stage_"):
            param.requires_grad = False
    for name, param in gen.named_parameters():
        if name.startswith(f"stage_{stage_id}"):
            param.requires_grad = True

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        x_scaled = bicubic_downsample(x, 2 ** (2-stage_id))
        y = bicubic_downsample(y, 2 ** (2-stage_id))
        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)[stage_id][stage_id]
            # print(y_fake.shape)
            D_real = disc(x_scaled, y)
            # print(x_.shape)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x_scaled, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x_scaled, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
    return disc, opt_disc

def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)[-1][-1]
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def charbonnier_loss(x, y):
    diff = x - y
    loss = torch.sum(torch.sqrt(diff * diff + 1e-6))
    return loss 

def main():
    # disc = Discriminator(in_channels=3).to(config.DEVICE)
    # gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    
    gen = MAXIM_dns_3s(features=12, depth=3, ).to(config.DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    # L1_LOSS = nn.L1Loss()
    L1_LOSS = charbonnier_loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for stage_id in range(3):
        for epoch in range(config.NUM_EPOCHS):
            disc, opt_disc =  train_stage(
                stage_id, gen, train_loader, opt_gen, L1_LOSS, BCE, g_scaler,
            )

            if config.SAVE_MODEL and epoch % 5 == 0:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

            save_some_examples(stage_id, gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()
