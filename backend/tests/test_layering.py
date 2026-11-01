"""分层约束：用 AST 扫 import，钉住能力层 / 流水线层 / 接口层的依赖方向。

``app/`` 内只拦跨包 import；``eval/`` 是独立的离线评测包，只允许依赖能力层与 infra，
不得反向 import 运行时编排。``config`` / ``core`` / ``trace`` 是横切包，任何层都可依赖。
包内互相引用（如 ``eval.rag.rag`` 之间）不受限制。
"""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = BACKEND_ROOT / "app"
EVAL_ROOT = BACKEND_ROOT / "eval"

# app/ 左侧包不得 import 右侧包：能力层不依赖流水线，流水线之间互不依赖，接口层只经服务层
FORBIDDEN = {
    "retrieval": {"agent", "ingestion", "services", "api"},
    "intent": {"agent", "ingestion", "services", "api"},
    "agent": {"ingestion", "services", "api"},
    "ingestion": {"agent", "services", "api"},
    "services": {"api"},
    "api": {"agent", "ingestion"},
    "infra": {"retrieval", "intent", "agent", "ingestion", "services", "api"},
    "ml": {"retrieval", "intent", "agent", "ingestion", "services", "api"},
}

# eval/ 只读能力层与 infra，不得 import 任何运行时编排包
EVAL_FORBIDDEN = {"agent", "ingestion", "services", "api"}


def _imported_modules(path: Path) -> set[str]:
    """收集文件里所有 ``app.*`` 导入的绝对模块名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return {m for m in modules if m.startswith("app.")}


def _collect_violations(root: Path, package: str, forbidden: set[str]) -> list[str]:
    package_dir = root / package
    if not package_dir.is_dir():
        return []
    violations: list[str] = []
    for path in package_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for module in _imported_modules(path):
            target = module.split(".")[1]
            if target in forbidden:
                violations.append(f"{path.relative_to(BACKEND_ROOT)}: {package} -> {module}")
    return violations


def test_app_packages_follow_layer_direction() -> None:
    violations: list[str] = []
    for package, forbidden in FORBIDDEN.items():
        violations += _collect_violations(APP_ROOT, package, forbidden)

    assert not violations, "跨层 import 违反依赖方向：\n" + "\n".join(violations)


def test_eval_does_not_import_runtime_pipelines() -> None:
    violations = _collect_violations(EVAL_ROOT, ".", EVAL_FORBIDDEN)

    assert not violations, "eval 反向依赖运行时编排：\n" + "\n".join(violations)
