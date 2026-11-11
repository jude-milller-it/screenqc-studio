LOCAL CHECKPOINTS -- excluded from Git by .gitignore

Required for the three models:
  deeplab_scheme3.pth
  segformer_mit_b2.pth
  fdsnet_phone.pth

For the separately trained DeepLabV3+ baseline:
  history/logs_baseline/best_epoch_weights.pth

Optional additional DeepLab stages:
  history/logs_loss_opt/best_epoch_weights.pth
  history/logs_cbam/best_epoch_weights.pth

Double-click 01_MIGRATE_ALL.cmd after unpacking beside your older project folders.
The script COPIES these files; originals are never removed.
