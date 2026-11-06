#!/usr/bin/env python3
"""Synthetic-only compatibility smoke test for the pinned MantisV2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_REVISION = "99fe0f548960e272fbfa4b82fd9b5b5956779dfd"
EXPECTED_MODEL_SHA256 = "49d46d9a49cccdc87c46f4e0088fa52c0a6ef7eb4c13de5cc9815426b7b17ab1"
EXPECTED_EMBEDDING_DIM = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_network(checkpoint: Path, device: str) -> torch.nn.Module:
    from mantis.architecture import MantisV2

    network = MantisV2(
        device=device,
        return_transf_layer=2,
        output_token="combined",
    )
    network = network.from_pretrained(str(checkpoint))
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def infer(network: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = network(inputs)
    assert outputs.shape == (inputs.shape[0], EXPECTED_EMBEDDING_DIM)
    assert torch.isfinite(outputs).all().item()
    return outputs


def run_device(checkpoint: Path, device: str) -> dict[str, Any]:
    network = load_network(checkpoint, device)
    generator = torch.Generator(device=device).manual_seed(5)
    multichannel = torch.randn(2, 25, 512, generator=generator, device=device)
    single_channel = multichannel[:, :1, :]
    before = [parameter.detach().clone() for parameter in network.parameters()]

    direct_error = None
    try:
        infer(network, multichannel)
    except RuntimeError as error:
        direct_error = str(error).splitlines()[0]
    assert direct_error is not None

    first = infer(network, single_channel)
    second = infer(network, single_channel.clone())
    assert torch.equal(first, second)

    from mantis.trainer import MantisTrainer

    trainer = MantisTrainer(device=device, network=network)
    multichannel_embedding = trainer.transform(
        multichannel,
        three_dim=True,
        to_numpy=False,
    )
    assert multichannel_embedding.shape == (2, 25, EXPECTED_EMBEDDING_DIM)
    assert torch.isfinite(multichannel_embedding).all().item()
    assert not any(parameter.grad is not None for parameter in network.parameters())
    after = [parameter.detach() for parameter in network.parameters()]
    assert all(torch.equal(left, right) for left, right in zip(before, after))

    return {
        "device": device,
        "direct_multichannel_shape": list(multichannel.shape),
        "direct_multichannel_rejected": True,
        "direct_multichannel_error": direct_error,
        "trainer_multichannel_output_shape": list(multichannel_embedding.shape),
        "trainer_multichannel_finite": True,
        "single_channel_input_shape": list(single_channel.shape),
        "single_channel_output_shape": list(first.shape),
        "finite": True,
        "repeated_inference_equal": True,
        "parameters_unchanged": True,
        "training_gradients_absent": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    model_file = args.checkpoint / "model.safetensors"
    assert model_file.is_file(), f"missing checkpoint: {model_file}"
    model_hash = sha256(model_file)
    assert model_hash == EXPECTED_MODEL_SHA256

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    results = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "model_sha256": model_hash,
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "return_transf_layer": 2,
        "output_token": "combined",
        "cpu": run_device(args.checkpoint, "cpu"),
    }
    if torch.cuda.is_available():
        results["cuda"] = run_device(args.checkpoint, "cuda")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
