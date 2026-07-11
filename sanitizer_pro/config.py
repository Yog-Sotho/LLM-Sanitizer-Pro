"""Configuration loading, merging, and custom script loading."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple
import re

from sanitizer_pro.utils import ConfigurationError

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

def load_config_file(config_path: str) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists(): raise ConfigurationError(f"Config file not found: {config_path}")
    raw = p.read_text(encoding='utf-8')
    if p.suffix.lower() in {'.yaml', '.yml'}:
        if not YAML_AVAILABLE: raise ConfigurationError("YAML config requires: pip install pyyaml")
        return yaml.safe_load(raw) or {}
    return json.loads(raw)

def collect_explicit_args(parser: argparse.ArgumentParser) -> Set[str]:
    """Safely detect explicit CLI args without mutating the original parser."""
    parser_copy = copy.deepcopy(parser)
    for action in parser_copy._actions:
        action.default = argparse.SUPPRESS
    explicit_ns, _ = parser_copy.parse_known_args()
    return set(vars(explicit_ns).keys())

def apply_config_to_args(args: argparse.Namespace, config: Dict[str, Any], explicit_args: Set[str]) -> None:
    current = vars(args)
    for key, value in config.items():
        if key in current and key not in explicit_args:
            setattr(args, key, value)

def load_quality_script(path: str) -> Callable[[Dict[str, Any]], bool]:
    import logging
    logging.warning(f"Loading custom quality script: {path}. Execute only trusted code.")
    spec = importlib.util.spec_from_file_location("_user_quality", path)
    if spec is None or spec.loader is None: raise ImportError(f"Cannot load: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'quality_check'): raise AttributeError(f"{path} must define quality_check")
    return module.quality_check

def load_custom_pii_patterns(path: str) -> List[Tuple[re.Pattern[str], str, str]]:
    entries = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(entries, list): raise ConfigurationError(f"Custom PII file must be JSON array: {path}")
    patterns = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or 'pattern' not in e or 'token' not in e:
            raise ConfigurationError(f"PII entry #{i} must be an object with 'pattern' and 'token': {path}")
        try:
            compiled = re.compile(e['pattern'], re.IGNORECASE)
        except re.error as exc:
            raise ConfigurationError(f"Invalid regex in PII entry #{i} ({e['pattern']!r}): {exc}") from None
        patterns.append((compiled, e['token'], e.get('kind', 'custom')))
    return patterns

def load_field_config(path: str) -> List[Dict[str, Any]]:
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config, list):
        raise ConfigurationError(f"Field config must be a JSON array: {path}")
    valid = {'rename', 'drop', 'pii_only', 'no_clean'}
    for i, e in enumerate(config):
        if not isinstance(e, dict) or 'field' not in e or 'action' not in e:
            raise ConfigurationError(f"Field config entry #{i} must have 'field' and 'action'.")
        if e['action'] not in valid:
            raise ConfigurationError(f"Invalid action in entry #{i}: {e['action']} (valid: {sorted(valid)})")
        if e['action'] == 'rename' and 'to' not in e:
            raise ConfigurationError(f"Rename entry #{i} for '{e['field']}' is missing 'to'.")
    return config

def build_field_ops(config: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Set[str], Set[str], Set[str]]:
    renames, drops, pii_only, no_clean = {}, set(), set(), set()
    for e in config:
        f, act = e['field'], e['action']
        if act == 'rename': renames[f] = e['to']
        elif act == 'drop': drops.add(f)
        elif act == 'pii_only': pii_only.add(f)
        elif act == 'no_clean': no_clean.add(f)
    if renames:
        pii_only = {renames.get(f, f) for f in pii_only}
        no_clean = {renames.get(f, f) for f in no_clean}
    return renames, drops, pii_only, no_clean
