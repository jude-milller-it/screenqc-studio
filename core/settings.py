"""Portable project configuration. Only local.json is user-editable."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    'dataset': 'data/MSD',
    'weights': {
        'deeplab': 'checkpoints/deeplab_scheme3.pth',
        'segformer': 'checkpoints/segformer_mit_b2.pth',
        'fdsnet': 'checkpoints/fdsnet_phone.pth',
    },
    'deeplab_scheme': 3,
    'fdsnet_stain_scale': 1.10,
    'ui_scale': 110,
    'ai': {'provider':'DeepSeek','model':'deepseek-flash','base':'https://api.deepseek.com'},
    'deeplab_variant_weights': {
        '0':'checkpoints/history/logs_baseline/best_epoch_weights.pth',
        '1':'checkpoints/history/logs_loss_opt/best_epoch_weights.pth',
        '2':'checkpoints/history/logs_cbam/best_epoch_weights.pth',
    },
    'segformer_oil_delta': 0.30,
    'segformer_oil_margin': 0.0,
    'outputs': 'outputs',
    'language': 'zh',
    'preferred_report': 'standard',
}


def get_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    local = ROOT / 'config' / 'local.json'
    if local.is_file():
        data = json.loads(local.read_text(encoding='utf-8-sig'))
        for k, v in data.items():
            if k == 'weights' and isinstance(v, dict):
                cfg['weights'].update(v)
            else:
                cfg[k] = v
    return cfg


def path_value(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def model_weights(model: str, cfg: dict | None = None) -> Path:
    conf = cfg or get_config()
    key = 'segformer' if model.startswith('segformer') else ('fdsnet' if model.startswith('fdsnet') else model)
    if model=='deeplab_base':
        return path_value(conf['deeplab_variant_weights']['0'])
    # V4 separate immutable keys for baseline versus fully trained CNN.
    # The old UI mutable deeplab_scheme must never silently replace full weights.
    return path_value(conf['weights'][key])


def save_config(cfg: dict) -> Path:
    path = ROOT / 'config' / 'local.json'
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    return path
