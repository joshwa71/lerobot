#!/usr/bin/env python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging

from lerobot.configs import parser
from lerobot.meta.configs import MetaTrainConfig
from lerobot.meta.engine import MetaEngine
from lerobot.utils.utils import init_logging


@parser.wrap()
def meta_train(cfg: MetaTrainConfig):
    cfg.validate()
    cfg_dict = cfg  # draccus already validated nested configs
    logging.info("Starting meta-training")
    engine = MetaEngine(cfg_dict)
    # Use CLI-configured steps, batch_size, log_freq, save_freq
    engine.train(
        total_outer_steps=cfg.steps,
        batch_size=cfg.batch_size,
        log_freq=cfg.log_freq,
        save_freq=cfg.save_freq,
    )


def main():
    init_logging()
    meta_train()


if __name__ == "__main__":
    main()
