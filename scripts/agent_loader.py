"""
Dynamic agent loader — load any Agent class from a .py file at runtime.
Used by run_local_match.py and estimate_rankings.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_agent_instance(agent_path: str, agent_id: int):
    """
    Dynamically load an Agent class from agent_path and instantiate it.

    Adds the agent's parent directory to sys.path so helper modules
    bundled alongside agent.py (e.g. model.onnx loader) can be imported.

    Args:
        agent_path: absolute or relative path to agent.py
        agent_id:   player index to pass to Agent.__init__

    Returns:
        Instantiated agent object with .act(obs) method
    """
    agent_dir = str(Path(agent_path).parent)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    spec = importlib.util.spec_from_file_location("agent", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec: {agent_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["agent"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("agent", None)
        raise

    agent_cls = getattr(module, "Agent", None)
    if agent_cls is None or not isinstance(agent_cls, type):
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr_name.endswith("Agent"):
                agent_cls = attr
                break
    if agent_cls is None:
        raise AttributeError(
            f"No Agent class found in {agent_path}. "
            "Expected a class named 'Agent' or ending with 'Agent'."
        )

    try:
        return agent_cls(agent_id)
    except TypeError:
        return agent_cls()
