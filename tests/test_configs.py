"""Every experiment config composes, and every agent config matches its agent's constructor."""

import inspect
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SCRIPTS = sorted(p.relative_to(CONFIG_DIR).with_suffix("").as_posix() for p in (CONFIG_DIR / "scripts").rglob("*.yaml"))


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_config_composes(name):
    with initialize_config_dir(str(CONFIG_DIR), version_base="1.1"):
        cfg = compose(name)
    assert "seed" in cfg and "env" in cfg


@pytest.mark.parametrize("name", [n for n in SCRIPTS if "pointmaze" not in n and "offline" not in n and "finetune" not in n])
def test_pretraining_agent_config_matches_constructor(name):
    with initialize_config_dir(str(CONFIG_DIR), version_base="1.1"):
        cfg = compose(name)
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=False)
    cls = get_class(agent_cfg.pop("_target_"))
    parameters = set(inspect.signature(cls.__init__).parameters) - {"self"}
    unknown = set(agent_cfg) - parameters
    assert not unknown, f"{name}: agent config keys not accepted by {cls.__name__}: {sorted(unknown)}"
    required = {p for p, v in inspect.signature(cls.__init__).parameters.items()
                if v.default is inspect.Parameter.empty and p != "self"}
    missing = required - set(agent_cfg)
    assert not missing, f"{name}: required constructor arguments missing from config: {sorted(missing)}"


def test_pointmaze_agent_config_matches_constructor():
    from agent.rover_subspace import SubspaceRoverAgent
    with initialize_config_dir(str(CONFIG_DIR), version_base="1.1"):
        cfg = compose("scripts/pointmaze/umaze_states")
    parameters = set(inspect.signature(SubspaceRoverAgent.__init__).parameters) - {"self"}
    assert set(cfg.agent) <= parameters, sorted(set(cfg.agent) - parameters)
    # Arguments the entry point supplies from the environment / top-level config.
    required = {p for p, v in inspect.signature(SubspaceRoverAgent.__init__).parameters.items()
                if v.default is inspect.Parameter.empty and p != "self"}
    assert required - set(cfg.agent) <= {"obs_dim", "n_actions", "discount"}
