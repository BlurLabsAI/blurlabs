import json
from dataclasses import dataclass
from pathlib import Path


_RAM2_REPO_ID = "blurlabs/ram2"
_RAM2_CONFIG_FILE = "config.json"


@dataclass(frozen=True)
class RAM2Config:
    checkpoint: str
    features: tuple[int, ...]
    block_numbers: tuple[int, ...]
    supported_channels: tuple[int, ...]
    unrolled_steps: int
    max_iter: int
    torch_dtype: str = "float32"

    @classmethod
    def from_json(cls, path: str) -> "RAM2Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (data.get("model_type"), data.get("format_version")) != ("ram2", 1):
            raise ValueError("expected a RAM2 config with format_version 1")
        architecture = data["architecture"]
        return cls(
            checkpoint=data["checkpoint"],
            features=tuple(architecture["features"]),
            block_numbers=tuple(architecture["block_numbers"]),
            supported_channels=tuple(architecture["supported_channels"]),
            unrolled_steps=architecture["unrolled_steps"],
            max_iter=data["inference_defaults"]["max_iter"],
            torch_dtype=data.get("torch_dtype", "float32"),
        )


DEFAULT_RAM2_CONFIG = RAM2Config(
    checkpoint="ram2.pth",
    features=(88, 176, 344, 688, 1376),
    block_numbers=(2, 4, 6, 10, 12),
    supported_channels=(3, 1, 2),
    unrolled_steps=10,
    max_iter=30,
)


def resolve_ram2_pretrained(
    default_config: RAM2Config, pretrained: str | None
) -> tuple[RAM2Config, str | None]:
    if pretrained != "download":
        return default_config, pretrained

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        raise ImportError(
            "to download the RAM2 weights, huggingface_hub is required. Install it using "
            "`pip install huggingface_hub`."
        ) from None

    try:
        config_path = hf_hub_download(_RAM2_REPO_ID, _RAM2_CONFIG_FILE)
        config = RAM2Config.from_json(config_path)
        checkpoint = hf_hub_download(_RAM2_REPO_ID, config.checkpoint)
    except GatedRepoError:
        raise RuntimeError(
            "To download the model, you need to first request access at https://huggingface.co/blurlabs/ram2, "
            "then run `hf auth login` or set HF_TOKEN."
        ) from None
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid RAM2 config in {_RAM2_REPO_ID}/{_RAM2_CONFIG_FILE}: {error}") from error

    return config, checkpoint
