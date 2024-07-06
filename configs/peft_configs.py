

def get_delta_key(delta_type):
    delta_keys = {
        "fine-tuning": "",
        "prefix": "num_virtual_tokens",
        "bitfit": "",
        "lora": "lora_rank",
        "adapter": "bottleneck_dim",
        "emulator": ""
    }
    delta_keys_abb = {
        "fine-tuning": "ft",
        "emulator": "",
        "prefix": "ptn",
        "bitfit": "",
        "lora": "la",
        "adapter": "dim"
    }
    return delta_keys[delta_type], delta_keys_abb[delta_type]
