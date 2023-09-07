import chex
chex.disable_asserts()

from src.config import CoderConfig
from src.runner import Runner


def main(config):
    runner = Runner(config)
    runner.run_replay_collection()
    if config.supervised:
        runner.run_supervised()
    else:
        runner.run_byol()
    runner.run_drq()


if __name__ == '__main__':
    cfg = CoderConfig.from_entrypoint()
    main(cfg)
