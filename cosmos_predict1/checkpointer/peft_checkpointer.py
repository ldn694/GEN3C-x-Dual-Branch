# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Any, Set

import torch

from cosmos_predict1.checkpointer.ddp import Checkpointer as DDPCheckpointer
from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.model import Model

# Substrings that identify a LoRA parameter, matching the module names built in
# cosmos_predict1/diffusion/training/utils/peft/lora_net.py (`self.net = nn.Sequential(down, up)`)
# and detected the same way in get_all_lora_params().
_LORA_KEY_MARKERS = ("lora.net.0", "lora.net.1")


class Checkpointer(DDPCheckpointer):
    """
    Checkpointer class for PEFT in distributed training. This class is similar to the DDP checkpointer,
    with the exception that the `broadcast_via_filesystem` functionality is not supported, and it supports
    loading pre-trained model without any postfix.

    Unlike the base DDP checkpointer, saved "model" checkpoints only contain LoRA parameters (a few
    tens of MB instead of the full base model). Resuming therefore always reloads the frozen base
    weights from `checkpoint.load_path` first, then overlays the LoRA weights (plus optimizer/
    scheduler/trainer state) from the local run's latest checkpoint, if any.

    Note:
    - Fully Sharded Data Parallelism (FSDP) is not supported by this checkpointer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.broadcast_via_filesystem:
            raise ValueError("self.broadcast_via_filesystem=False is not implemented for PEFT checkpointer.")

    def add_type_postfix_to_checkpoint_path(self, key: str, checkpoint_path: str, model: Model) -> str:
        """
        Overwrite the `add_type_postfix_to_checkpoint_path` function of the base class (DDP checkpointer)
        to load pre-trained model without any postfix.
        """
        checkpoint_path = super().add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
        checkpoint_path = checkpoint_path.replace("model_model.pt", "model.pt")
        return checkpoint_path

    def generate_save_state_dict(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> dict[str, Any] | None:
        """
        Same as the DDP checkpointer, except the "model" entry is filtered down to LoRA parameters
        only. The frozen base weights are never modified during LoRA training, so they don't need to
        be re-saved every checkpoint - they're reloaded from `checkpoint.load_path` on resume instead.
        """
        state_dict = super().generate_save_state_dict(model, optimizer, scheduler, grad_scaler, iteration)
        if state_dict and state_dict.get("model"):
            full_model_state = state_dict["model"]
            lora_state = {k: v for k, v in full_model_state.items() if any(m in k for m in _LORA_KEY_MARKERS)}
            if not lora_state:
                raise ValueError("No LoRA parameters found in model.state_dict(); refusing to save an empty checkpoint.")
            if "trained_data_record" in full_model_state:
                lora_state["trained_data_record"] = full_model_state["trained_data_record"]
            state_dict["model"] = lora_state
        return state_dict

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """
        Two-step load:
        1. Base weights from `checkpoint.load_path` (non-strict: that checkpoint has no LoRA keys).
        2. If a local `latest_checkpoint.txt` exists (i.e. this job has saved LoRA checkpoints
           before), overlay LoRA weights + optimizer/scheduler/trainer state on top (also
           non-strict: the local checkpoint has no base-model keys).

        Returns:
            iteration (int): the iteration number to resume from (0 if no local checkpoint yet).
        """
        self.callbacks.on_load_checkpoint_start(model)

        if self.load_path:
            self._check_checkpoint_exists(self.load_path)
            base_state = self.load_broadcast_state_dict(self.load_path, model, {"model"})
            log.info("- Loading base pretrained weights...")
            model_load_info = model.load_state_dict(base_state["model"], strict=False)
            log.info(f"\t {model_load_info}")

        iteration = 0
        latest_checkpoint_file = self._read_latest_checkpoint_file()
        if latest_checkpoint_file is not None:
            checkpoint_path = os.path.join(self.load_dirname, latest_checkpoint_file)
            self._check_checkpoint_exists(checkpoint_path)
            resume_keys = set(self.KEYS_TO_SAVE) - set(self.keys_not_to_resume)
            state_dict = self.load_broadcast_state_dict(checkpoint_path, model, resume_keys)

            if "trainer" in state_dict:
                trainer_state = state_dict["trainer"]
                log.info("- Loading the gradient scaler...")
                grad_scaler.load_state_dict(trainer_state["grad_scaler"])
                self.callbacks.on_load_checkpoint(model, state_dict=trainer_state)
                iteration = trainer_state["iteration"]
            if "optim" in state_dict:
                assert optimizer
                log.info("- Loading the optimizer...")
                optimizer.load_state_dict(state_dict["optim"])
            if "scheduler" in state_dict:
                assert scheduler
                log.info("- Loading the scheduler...")
                scheduler.load_state_dict(state_dict["scheduler"])
                scheduler.last_epoch = iteration
            if "model" in state_dict:
                log.info("- Loading LoRA weights...")
                model_load_info = model.load_state_dict(state_dict["model"], strict=False)
                log.info(f"\t {model_load_info}")
            self.print(f"Resumed LoRA checkpoint from {checkpoint_path} at iteration {iteration}")
        elif self.load_path:
            self.print(f"Loaded base pretrained weights from {self.load_path}; starting LoRA training from scratch.")
        else:
            log.info("Training from scratch.")

        torch.cuda.empty_cache()
        self.callbacks.on_load_checkpoint_end(model)
        return iteration

    def load_broadcast_state_dict(self, checkpoint_path: str, model: Model, resume_keys: Set) -> dict[str, Any]:
        """
        Load state_dict and broadcast for PEFT checkpointer.

        This function is identical to the `load_broadcast_state_dict` function of the base class (DDP checkpointer),
        with the exception that the `broadcast_via_filesystem` functionality is not supported.

        Args:
            checkpoint_path (str): The base path of the checkpoint.
            model (Model): The model being loaded.
            resume_keys (Set): Set of keys to resume from the checkpoint.

        Returns:
            dict[str, Any]: A dictionary containing the loaded state for each resumed key.
        """
        state_dict = {}
        sorted_resume_keys = sorted(resume_keys)
        # Step 1: Download checkpoints for every GPU of DDP-rank 0 and CP-rank 0.
        if self.rank_dp_w_cp == 0:
            for key in sorted_resume_keys:
                _ckpt_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
                local_cache_path = os.path.join(self.load_dirname, os.path.basename(_ckpt_path))
                if os.path.exists(local_cache_path):
                    # If the local checkpoint exists, we can directly load it
                    self.print(f"Checkpoint is already in local cache: {local_cache_path}. Loading...")
                    _state_dict = torch.load(
                        local_cache_path, map_location=lambda storage, loc: storage, weights_only=False
                    )
                else:
                    # Pre-trained model is not in local cache, so we need to load it from the checkpoint path
                    self.print(f"Loading checkpoint from: {_ckpt_path}")
                    _state_dict = torch.load(_ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
                state_dict[key] = _state_dict

        # Ensure all ranks wait for the download to complete
        distributed.barrier()

        # Step 2: Broadcast checkpoint data
        log.info(
            "Start broadcasting checkpoint from the source rank to all other ranks in the same DDP group.",
            rank0_only=True,
        )
        for key in sorted_resume_keys:
            if self.broadcast_via_filesystem:
                # Load the checkpoint from the local filesystem for other ranks
                if self.rank_dp_w_cp != 0:
                    _ckpt_path = self.add_type_postfix_to_checkpoint_path(key, checkpoint_path, model)
                    local_cache_path = os.path.join(self.load_dirname, os.path.basename(_ckpt_path))
                    if os.path.exists(local_cache_path):
                        self.print(f"Loading checkpoint from: {local_cache_path}")
                        state_dict[key] = torch.load(
                            local_cache_path, map_location=lambda storage, loc: storage, weights_only=False
                        )
                    else:
                        self.print(f"Loading checkpoint from: {_ckpt_path}")
                        state_dict[key] = torch.load(
                            _ckpt_path, map_location=lambda storage, loc: storage, weights_only=False
                        )

            else:
                raise ValueError("self.broadcast_via_filesystem=False is not implemented for PEFT checkpointer.")

        return state_dict
