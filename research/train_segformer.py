"""
SegFormer training script for industrial defect segmentation.

Key features:
  - SegFormer (MiT encoder + MLP decoder) — Transformer for long-range scratch continuity
  - Dice + Focal + Lovász + Boundary combined loss
  - OHEM: 100% defect pixels + top-10% background
  - Copy-Paste augmentation + biased cropping (via IndustrialDataset)
  - Mixed precision (fp16) + cosine LR schedule

Usage:
    python train_segformer.py

配置总开关: 修改 scheme 即可切换 MiT 变体大小
"""
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.segformer import SegFormer, SEGFORMER_VARIANTS
from nets.segformer_training import SegFormerLoss, weights_init_segformer
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader_industrial import IndustrialDataset, industrial_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn


if __name__ == "__main__":
    # ====================== 实验方案一键切换总开关 ======================
    # scheme: 选择 MiT backbone 变体
    #   0 = MiT-b0 (轻量, ~3.7M params, 最快)
    #   1 = MiT-b1 (~13.7M params)
    #   2 = MiT-b2 (~24.7M params, 推荐 — 精度/速度平衡)
    #   3 = MiT-b3 (~44.6M params, 最强 ★推荐★)
    #   4 = MiT-b4 (~61.4M params, 需要更大显存)
    #   5 = MiT-b5 (~81.9M params, 需要更大显存)
    # ==================================================================
    scheme = 2  # 默认 MiT-b2，RTX 3050 4GB 稳妥；显存够可改 3

    variant_map = {i: v for i, v in enumerate(SEGFORMER_VARIANTS)}
    variant = variant_map.get(scheme, 'mit_b2')

    # ---------------------------------#
    #   基础配置
    # ---------------------------------#
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = True  # 混合精度，4GB 显存必开

    num_classes = 4
    # 输入尺寸: 512x512 (平衡速度与精度)
    input_shape = [512, 512]
    # 数据集
    VOCdevkit_path = 'D:/AI/dataset/MobilePhoneScreenSurfaceDefectSegmentation/MSD'
    # 日志/权重保存目录
    save_dir = f"logs_segformer_{variant}"

    # ---------------------------------#
    #   训练超参
    # ---------------------------------#
    Init_Epoch = 0
    UnFreeze_Epoch = 200        # 总训练轮数（从头训练需要更多轮）
    Freeze_Epoch = 30           # 冻结 encoder 的轮数
    Freeze_Train = False        # 从头训练不冻结，全模型一起学
    Freeze_batch_size = 8       # 冻结阶段 batch (仅训练 decoder)
    Unfreeze_batch_size = 4     # 解冻后 batch (全模型)

    # 学习率 (AdamW, Transformer 用较小 LR)
    Init_lr = 6e-5
    Min_lr = Init_lr * 0.01
    # 注意：自适应缩放会按 batch_size/nbs 调整 LR。
    # 设 lr_limit_min 为 6e-5 确保实际 LR 不低于此值。
    lr_limit_min_override = 6e-5  # 强制最低 LR，不被自适应缩放压低
    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 0.01         # AdamW weight decay

    lr_decay_type = 'cos'
    save_period = 5
    eval_flag = True
    eval_period = 5
    num_workers = 0             # Windows spawn 多进程 + Copy-Paste 大实例池 = 极慢，设 0

    # ---------------------------------#
    #   损失函数权重 (Phase 2: 平衡背景/缺陷)
    # ---------------------------------#
    loss_weights = {
        'w_dice': 2.0,
        'w_focal': 1.0,
        'w_lovasz': 1.0,
        'w_boundary': 0.15,    # 降低：不强迫模型关注微弱边缘
    }
    # 类别权重 — 背景从 0.1 提到 0.35
    cls_weights = np.array([0.35, 0.7, 1.1, 1.0], dtype=np.float32)
    ohem_bg_ratio = 0.25        # 背景取 top 25% 参与 OHEM

    # ---------------------------------#
    #   Copy-Paste & 数据增强
    # ---------------------------------#
    use_copy_paste = True
    cp_num_defects = (1, 2)     # paste 1-2 defects (reduced from 3-5)
    cp_num_scratches = (0, 1)   # paste 0-1 scratches (reduced from 2-3)
    biased_crop_prob = 0.3      # 30% defect-centered crop (was 50%: more random bg)

    # ---------------------------------#
    #   Resume / Pretrained
    # ---------------------------------#
    pretrained = True
    model_path = ''  # 单独加载权重（不含optimizer状态）
    resume_checkpoint = os.path.join(save_dir, 'checkpoint.pth')  # 断点续训，自动检测，存在即续
    # ======================================== #
    #   以下为训练流程，通常无需修改
    # ======================================== #

    seed_everything(seed)

    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"Gpu Device Count: {ngpus_per_node}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    print(f"[train_segformer] Using variant: {variant}, device: {device}")

    # ---- build model ----
    model = SegFormer(num_classes=num_classes, variant=variant, pretrained=pretrained)
    if not pretrained:
        weights_init_segformer(model.decoder)

    # ---- load checkpoint (resume) or standalone weights ----
    checkpoint_data = None
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f'\n{"="*60}')
        print(f'[Resume] Loading checkpoint: {resume_checkpoint}')
        checkpoint_data = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data['model'])
        Init_Epoch = checkpoint_data['epoch'] + 1  # 覆盖 Init_Epoch，后续所有 epoch==Init_Epoch 判断自动生效
        print(f'[Resume] Epoch {checkpoint_data["epoch"]} -> resume from {Init_Epoch}')
        print(f'[Resume] Loss={checkpoint_data.get("loss", "N/A"):.4f}, '
              f'ValLoss={checkpoint_data.get("val_loss", "N/A"):.4f}, '
              f'mIoU={checkpoint_data.get("miou", 0):.2f}%')
        print(f'{"="*60}\n')
    elif model_path and os.path.exists(model_path):
        print(f'Loading weights from {model_path}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device, weights_only=False)
        load_key, no_load_key = [], []
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(load_key)} keys, skipped {len(no_load_key)} keys.")
    else:
        print(f'\n[Fresh] No checkpoint found — starting fresh training from epoch 0\n')

    # ---- loss history ----
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        os.makedirs(log_dir, exist_ok=True)
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ---- amp scaler ----
    if fp16:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    model_train = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            model_train = model_train.cuda()

    # ---- loss function ----
    segformer_loss_fn = SegFormerLoss(
        num_classes=num_classes,
        cls_weights=torch.from_numpy(cls_weights).float(),
        **loss_weights,
        gamma=1.5,              # 降低聚焦强度
        ohem_bg_ratio=ohem_bg_ratio,
    )
    if Cuda:
        segformer_loss_fn = segformer_loss_fn.cuda()

    # ---- data loading ----
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            variant=variant, num_classes=num_classes, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr,
            optimizer_type=optimizer_type, lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers,
            num_train=num_train, num_val=num_val,
            copy_paste=use_copy_paste, ohem_bg_ratio=ohem_bg_ratio,
            **loss_weights,
        )

        wanted_step = 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(f"\n\033[1;33;44m[Warning] 总训练步长 {total_step} < 建议 {wanted_step}, "
                  f"建议设置总世代为 {wanted_epoch}\033[0m")

    # ---- train/val datasets ----
    train_dataset = IndustrialDataset(
        train_lines, input_shape, num_classes, train=True, dataset_path=VOCdevkit_path,
        copy_paste=use_copy_paste, cp_num_defects=cp_num_defects,
        cp_num_scratches=cp_num_scratches, biased_crop_prob=biased_crop_prob,
    )
    val_dataset = IndustrialDataset(
        val_lines, input_shape, num_classes, train=False, dataset_path=VOCdevkit_path,
        copy_paste=False,
    )

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size_per_gpu = Unfreeze_batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        batch_size_per_gpu = Unfreeze_batch_size
        shuffle = True

    # ---- LR scheduler ----
    from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr

    # ---- create DataLoaders (before loop, fixes resume bug) ----
    UnFreeze_flag = False
    if Freeze_Train:
        model.freeze_backbone()
        batch_size = Freeze_batch_size
        print(f"[Init] Freeze phase: batch_size={batch_size}, training decoder only")
    else:
        batch_size = Unfreeze_batch_size
        UnFreeze_flag = True  # no freeze phase needed
        print(f"[Init] Training from scratch: batch_size={batch_size}, full model")

    if distributed:
        batch_size_per_gpu = batch_size // ngpus_per_node

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size_per_gpu,
                     num_workers=num_workers, pin_memory=True, drop_last=True,
                     collate_fn=industrial_dataset_collate, sampler=train_sampler,
                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size_per_gpu,
                         num_workers=num_workers, pin_memory=True, drop_last=True,
                         collate_fn=industrial_dataset_collate, sampler=val_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

    # ---- eval callback (create ONCE, not every epoch) ----
    if local_rank == 0:
        eval_callback = EvalCallback(model, input_shape, num_classes, val_lines,
                                     VOCdevkit_path, log_dir, Cuda,
                                     eval_flag=eval_flag, period=eval_period)
    else:
        eval_callback = None

    # ---- training loop ----
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # ---- handle freeze -> unfreeze transition ----
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            print(f"\n[Epoch {epoch}] Unfreezing backbone...")
            model.unfreeze_backbone()
            batch_size = Unfreeze_batch_size
            UnFreeze_flag = True

            # rebuild dataloaders with new batch size
            if distributed:
                batch_size_per_gpu = batch_size // ngpus_per_node

            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size_per_gpu,
                             num_workers=num_workers, pin_memory=True, drop_last=True,
                             collate_fn=industrial_dataset_collate, sampler=train_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size_per_gpu,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 collate_fn=industrial_dataset_collate, sampler=val_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # ---- optimizer (AdamW for transformer) ----
        if epoch == Init_Epoch or (epoch == Freeze_Epoch and Freeze_Train):
            # adaptive LR based on batch size
            nbs = 4   # 以 bs=4 为基准，不再按 bs=16 缩放
            lr_limit_max = 1e-4
            lr_limit_min = lr_limit_min_override  # SegFormer 从头训练需要足够的 LR
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            if optimizer_type == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=Init_lr_fit,
                                        betas=(0.9, 0.999), weight_decay=weight_decay)
            elif optimizer_type == "adam":
                optimizer = optim.Adam(model.parameters(), lr=Init_lr_fit,
                                       betas=(momentum, 0.999), weight_decay=weight_decay)
            else:
                optimizer = optim.SGD(model.parameters(), lr=Init_lr_fit,
                                      momentum=momentum, nesterov=True, weight_decay=weight_decay)

            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            # restore optimizer & scaler from checkpoint
            if checkpoint_data is not None:
                try:
                    optimizer.load_state_dict(checkpoint_data['optimizer'])
                    print('[Resume] Optimizer state restored')
                except Exception as e:
                    print(f'[Resume] Optimizer restore failed (will use fresh): {e}')
                if scaler is not None and checkpoint_data.get('scaler') is not None:
                    try:
                        scaler.load_state_dict(checkpoint_data['scaler'])
                        print('[Resume] Scaler state restored')
                    except Exception:
                        pass
                checkpoint_data = None  # only restore once

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        # ---- one epoch ----
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法训练。请扩充数据集或减小 batch_size。")

        # ---- training iteration ----
        from tqdm import tqdm
        from utils.utils import get_lr
        from utils.utils_metrics import f_score

        model_train.train()
        total_loss = 0.0
        total_f_score = 0.0

        if local_rank == 0:
            pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}',
                        postfix=dict, mininterval=0.3)

        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch

            if Cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)

            optimizer.zero_grad()

            if not fp16:
                outputs = model_train(imgs)
                loss = segformer_loss_fn(outputs, labels)
                with torch.no_grad():
                    _f_score = f_score(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                with torch.amp.autocast('cuda'):
                    outputs = model_train(imgs)
                    loss = segformer_loss_fn(outputs, labels)
                    with torch.no_grad():
                        _f_score = f_score(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{
                    'total_loss': total_loss / (iteration + 1),
                    'f_score': total_f_score / (iteration + 1),
                    'lr': get_lr(optimizer),
                })
                pbar.update(1)

        if local_rank == 0:
            pbar.close()

        # ---- validation ----
        model_train.eval()
        val_loss = 0.0
        val_f_score = 0.0

        if local_rank == 0:
            pbar = tqdm(total=epoch_step_val, desc=f'Val {epoch + 1}/{UnFreeze_Epoch}',
                        postfix=dict, mininterval=0.3)

        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels = batch

            if Cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)

            with torch.no_grad():
                outputs = model_train(imgs)
                loss = segformer_loss_fn(outputs, labels)
                _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{
                    'val_loss': val_loss / (iteration + 1),
                    'f_score': val_f_score / (iteration + 1),
                    'lr': get_lr(optimizer),
                })
                pbar.update(1)

        if local_rank == 0:
            pbar.close()

        # ---- logging & saving ----
        avg_train_loss = total_loss / epoch_step
        avg_val_loss = val_loss / epoch_step_val

        if local_rank == 0:
            loss_history.append_loss(epoch + 1, avg_train_loss, avg_val_loss)
            eval_callback.on_epoch_end(epoch + 1, model_train)
            print(f'Epoch: {epoch+1}/{UnFreeze_Epoch}')
            print(f'Total Loss: {avg_train_loss:.3f} || Val Loss: {avg_val_loss:.3f}')

            # save checkpoint
            if (epoch + 1) % save_period == 0 or epoch + 1 == UnFreeze_Epoch:
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f'ep{epoch+1:03d}-loss{avg_train_loss:.3f}-val_loss{avg_val_loss:.3f}.pth'))

            if len(loss_history.val_loss) <= 1 or avg_val_loss <= min(loss_history.val_loss):
                print('Save best model to best_epoch_weights.pth')
                torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

            torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

            # ---- full checkpoint for resume ----
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None,
                'loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'miou': eval_callback.mious[-1] if eval_callback and eval_callback.mious else 0.0,
            }, os.path.join(save_dir, "checkpoint.pth"))

        if distributed:
            import torch.distributed as dist
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()
        print(f"\n{'='*60}")
        print(f"Training complete! Weights saved to: {save_dir}/")
        print(f"Best model: {save_dir}/best_epoch_weights.pth")
        print(f"{'='*60}")
