import argparse
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import monai
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from typing import Any, Iterable
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from dataset import LithosDataset

#%% set up model
class LithosSAM(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                multiple_pols=False,
                num_pols=0
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.num_pols = num_pols

        if multiple_pols:
            #! Get parameters of the first layer of the patch_embed
            pe_kernel_size = self.image_encoder.patch_embed.proj.kernel_size
            pe_stride = self.image_encoder.patch_embed.proj.stride
            pe_out_channels = self.image_encoder.patch_embed.proj.out_channels
            pe_padding = self.image_encoder.patch_embed.proj.padding
            original_model_state = self.state_dict()
            pe_state_dict = original_model_state[
                "image_encoder.patch_embed.proj.weight"
            ]
            # pre_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print(f"Original parameter: {pre_freeze}")

            #! Modify first layer of patch_embed
            num_int_channels = self.num_pols * 3
            new_pe_state_dict = pe_state_dict.repeat(1, self.num_pols, 1, 1)
            new_model_state_dict = self.state_dict()
            new_model_state_dict[
                "image_encoder.patch_embed.proj.weight"
            ] = new_pe_state_dict
            self.image_encoder.patch_embed.proj = nn.Conv2d(
                num_int_channels,
                pe_out_channels,
                kernel_size=pe_kernel_size,
                stride=pe_stride,
                padding=pe_padding,
            )
            self.load_state_dict(new_model_state_dict)

    def forward(self, image, im_point):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=im_point,
                boxes=None,
                masks=None,
            )
        
        mask_predictions, _ = model.mask_decoder(
            image_embeddings=image_embedding
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        return mask_predictions


def dice_score(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()

    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice

def main_worker(gpu, train_df, val_df, ngpus_per_node, args):

    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    image_dir = args.image
    mask_dir = args.mask
    sam_checkpoint_dir = args.checkpoint
    model_type = args.model_type
    num_pols = args.num_pols
    multiple_pols = args.multiple_pols
    model_save_path = os.path.join("runs", args.experiment_name)

    node_rank = init(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    if is_main_host:
        os.makedirs(save_path, exist_ok=True)
    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend = "nccl",
        init_method = args.init_method,
        rank = rank,
        world_size = world_size
    )

    #make model
    sam_model = sam_model_registry[args.model_type](
            checkpoint=args.sam_checkpoint_dir
        )
    model =  LithosSAM(
        image_encoder=sam_model.image_encoder, 
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder
    ).cuda()

    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0]/(1024**3), cuda_mem_info[1]/(1024**3)
    print(f'[RANK {rank}: GPU {gpu}] Total CUDA memory before DDP initialised: {total_cuda_mem} Gb')
    print(f'[RANK {rank}: GPU {gpu}] Free CUDA memory before DDP initialised: {free_cuda_mem} Gb')
    if rank % ngpus_per_node == 0:
        print("Before DDP initialization:")
        os.system('nvidia-smi')
    
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu], 
        output_device=gpu,
        gradient_as_bucket_view = True,
        find_unused_parameters = True,
        bucket_cap_mb = args.bucket_cap_mb
    )

    cuda_mem_info = torch.cuda.mem_get_info(gpu)
    free_cuda_mem, total_cuda_mem = cuda_mem_info[0]/(1024**3), cuda_mem_info[1]/(1024**3)
    print(f'[RANK {rank}: GPU {gpu}] Total CUDA memory after DDP initialised: {total_cuda_mem} Gb')
    print(f'[RANK {rank}: GPU {gpu}] Free CUDA memory after DDP initialised: {free_cuda_mem} Gb')
    if rank % ngpus_per_node == 0:
        print('After DDP initialization:')
        os.system('nvidia-smi')
    
    model.train()

    print('Number of total parameters: ', sum(p.numel() for p in model.parameters()))
    print('Number of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    trainable_parameters_sam_lithos = [
        "image_encoder.patch_embed.proj.weight",
        "image_encoder.patch_embed.proj.bias",
    ]
    for name, param in model.named_parameters():
        param.requires_grad = (
            True if name in trainable_parameters_sam_lithos else False
    )

    # Start training process 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    seg_loss = monai.losses.DiceCELoss(
        sigmoid=True, squared_pred=True, reduction="mean"
    )

    iter_num = 0
    losses = []
    best_loss = 1e10
    train_ds = LithosDataset(
        df=train_df,
        image_col=args.image_col,
        mask_col=args.mask_col,
        image_dir=image_dir,
        mask_dir=mask_dir,
        multiple_pols=multiple_pols,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    print('Number of training samples:', len(train_ds))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle = (train_sampler is None),
        num_workers = args.num_workers,
        pin_memory = True,
        sampler = train_sampler
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(rank, "=> loading checkpoint '{}'".format(args.resume))
            ## Map model to be loaded to specified single GPU
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(args.resume, map_location = loc)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(rank, "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        torch.distributed.barrier()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"[RANK {rank}: GPU {gpu}] Using AMP for training")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        train_dataloader.sampler.set_epoch(epoch)
        for step, (image, mask, _ , im_point) in enumerate(tqdm(train_dataloader, desc = f"[RANK {rank}: GPU {gpu}]")):
            optimizer.zero_grad()
            #boxes_np = boxes.detach().cpu().numpy()
            #image, gt2D = image.to(device), gt2D.to(device)
            image, mask = image.cuda(), mask.cuda()
            im_point[0] = im_point[0].cuda()
            im_point[1] = im_point[1].cuda()
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    lithosam_pred = model(image, im_point)
                    loss = seg_loss(lithosam_pred, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                lithosam_pred = model(image, im_point)
                loss = loss = seg_loss(lithosam_pred, mask)
                # Gradient accumulation
                if args.grad_acc_steps > 1:
                    loss = loss / args.grad_acc_steps  # normalize the loss because it is accumulated
                    if (step + 1) % args.grad_acc_steps == 0:
                        ## Perform gradient sync
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        ## Accumulate gradient on current node without backproping
                        with model.no_sync():
                            loss.backward() ## calculate the gradient only
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            
            if step>10 and step % 100 == 0:
                if is_main_host:
                    checkpoint = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}
                    torch.save(checkpoint, join(model_save_path, 'model_latest_step.pth'))

            epoch_loss += loss.item()
            iter_num += 1

            # if rank % ngpus_per_node == 0:
            #     print('\n')
            #     os.system('nvidia-smi')
            #     print('\n')

        # Check CUDA memory usage
        cuda_mem_info = torch.cuda.mem_get_info(gpu)
        free_cuda_mem, total_cuda_mem = cuda_mem_info[0]/(1024**3), cuda_mem_info[1]/(1024**3)
        print('\n')
        print(f'[RANK {rank}: GPU {gpu}] Total CUDA memory: {total_cuda_mem} Gb')
        print(f'[RANK {rank}: GPU {gpu}] Free CUDA memory: {free_cuda_mem} Gb')
        print(f'[RANK {rank}: GPU {gpu}] Used CUDA memory: {total_cuda_mem - free_cuda_mem} Gb')
        print('\n')

        epoch_loss /= step
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        # save the model checkpoint
        if is_main_host:
            checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, join(model_save_path, 'model_latest.pth'))
            
            ## save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(checkpoint, join(model_save_path, 'model_best.pth'))
        torch.distributed.barrier()

        # %% plot loss
        plt.plot(losses)
        plt.title('Dice Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.show() # comment this line if you are running on a server
        plt.savefig(join(model_save_path,'train_loss.png'))
        plt.close()



def main():
    # set up parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, required=True, help="Cross validation fold")

    parser.add_argument(
        "--multiple_pols",
        type=bool,
        default=False,
        help="Use multiple polarizations",
    )

    parser.add_argument(
        "--num_pols",
        type=int,
        default=20,
        help="Number of polarizations to use",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="/media/SSD7/LITHOS/COCO_ANNOTATIONS",
        help="Path to the CSV file",
    )

    parser.add_argument(
        "--image_col",
        type=str,
        default="Sec_patch",
        help="Name of the column on the dataframe that holds the image file names",
    )

    parser.add_argument(
        "--mask_col",
        type=str,
        default="Sec_patch",
        help="the name of the column on the dataframe that holds the mask file names",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/media/SSD7/LITHOS/SEMANTIC_SEGMENTATION_DATASET",
        help="Path to the input image directory",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="/media/SSD7/LITHOS/SAM_ANNOTATIONS",
        help="Path to the ground truth mask directory",
    )
    parser.add_argument(
        "--num_epochs", type=int, required=False, default=1, help="number of epochs"
    )
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument(
        "--lr", type=float, required=False, default=3e-4, help="learning rate"
    )
    parser.add_argument('-weight_decay', type=float, default=0.001,
        help='weight decay (default: 0.01)')
    parser.add_argument(
        "--batch_size", type=int, required=False, default=6, help="batch size"
    )
    parser.add_argument("--model_type", type=str, required=False, default="vit_b")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/SSD6/pruiz/LITHOS/segment-anything-lithos/segment_anything/models/sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint",
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Folder to save experiment"
    )
    parser.add_argument('--resume', type = str, default = '',
                    help="Resuming training from checkpoint")
    parser.add_argument('-use_amp', action='store_true', default=False, 
                    help='use amp')  
    ## Distributed training args
    parser.add_argument('--world_size', type=int, help='world size')
    parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
    parser.add_argument('--bucket_cap_mb', type = int, default = 25,
                        help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')
    parser.add_argument('--grad_acc_steps', type = int, default = 1,
                        help='Gradient accumulation steps before syncing gradients for backprop')
    parser.add_argument('--resume', type = str, default = '',
                        help="Resuming training from checkpoint")
    parser.add_argument('--init_method', type = str, default = "env://")


    args = parser.parse_args()

    try:
        if args.fold == 1:
            train_df = pd.read_csv(
                os.path.join(args.csv, f"Fold{str(args.fold)}_complete_info.csv")
            )
            val_df = pd.read_csv(os.path.join(args.csv, f"Fold2_complete_info.csv"))
        elif args.fold == 2:
            train_df = pd.read_csv(
                os.path.join(args.csv, f"Fold{str(args.fold)}_complete_info.csv")
            )
            val_df = pd.read_csv(os.path.join(args.csv, f"Fold1_complete_info.csv"))
    except FileNotFoundError:
        name_file = os.path.join(args.csv, f"Fold{str(args.fold)}.csv")
        print(f"{name_file} does not exist")

    print(f"[INFO] Starting training for {args.num_epochs} epochs ....")

    save_path = os.path.join("runs", args.experiment_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces")
    mp.spawn(main_worker, nprocs = ngpus_per_node, args=(train_df,val_df ,ngpus_per_node, args))


if __name__ == "__main__":
    main()
