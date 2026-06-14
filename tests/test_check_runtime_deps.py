import importlib.util
import sys
from pathlib import Path


def _load_check_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "check_runtime_deps.py"
    spec = importlib.util.spec_from_file_location("check_runtime_deps", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_defaults_to_base_profile():
    module = _load_check_module()

    args = module.parse_args([])

    assert args.profile == "base"
    assert args.require_cuda is False


def test_build_checks_for_rng_profile_includes_model_dependencies():
    module = _load_check_module()

    checks = module.build_checks("rng-cu121")

    names = [check.name for check in checks]
    assert "python" in names
    assert "torch" in names
    assert "torch_cuda" in names
    assert "pytorch3d" in names
    assert "grasp_nms" in names


def test_run_checks_reports_missing_path(tmp_path):
    module = _load_check_module()

    results = module.run_checks([module.path_exists_check("missing", tmp_path / "missing")])

    assert len(results) == 1
    assert results[0].ok is False
    assert "missing" in results[0].detail
