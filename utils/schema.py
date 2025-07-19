# utils/schema.py

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "data_dir": {"type": "string"},
        "checkpoint_dir": {"type": "string"},
        "pretrained_path": {"type": "string"},
        "resume": {"type": "string"},
        "batch_size": {"type": "integer"},
        "epochs": {"type": "integer"},
        "lr": {"type": "number"},
        "patience": {"type": "integer"},
        "image_size": {"type": "integer"},
        "num_workers": {"type": "integer"},
        "freeze_encoder": {"type": "boolean"},
        "use_wandb": {"type": "boolean"},
        "wandb_project": {"type": "string"},
        "notify_slack": {"type": "boolean"},
        "notify_email": {"type": "boolean"},
        "encoder": {"type": "object"},
        "decoder": {"type": "object"},
    },
    "required": ["data_dir", "checkpoint_dir", "pretrained_path", "batch_size", "epochs", "lr", "image_size", "encoder", "decoder"]
}
