# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore

from cosmos_predict1.diffusion.training.config.gen3c.experiment import register_experiments


def register_configs():
    cs = ConfigStore.instance()

    register_experiments(cs)
