"""分层约束：用 AST 扫 import，钉住能力层 / 流水线层 / 接口层的依赖方向。

只拦跨包 import；``config`` / ``core`` / ``trace`` 是横切包，任何层都可依赖。
包内互相引用（如 ``app.eval.rag.rag``）不受限制。
"""

from __future__ import annotations

import ast
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1] / "app"

# 左侧包不得 import 右侧包：能力层不依赖流水线，流水线之间互不依赖，接口层只经服务层
FORBIDDEN = {
    "retrieval": {"study_agent", "textbook_flow", "services", "api", "eval"},
    "intent": {"study_agent", "textbook_flow", "services", "api", "eval"},
    "study_agent": {"textbook_flow", "services", "api", "eval"},
    "textbook_flow": {"study_agent", "services", "api", "eval"},
    "services": {"api", "eval"},
    "api": {"study_agent", "textbook_flow", "eval"},
    "eval": {"study_agent", "textbook_flow", "services", "api"},
    "clients": {"retrieval", "intent", "study_agent", "textbook_flow", "services", "api", "eval"},
    "ml": {"retrieval", "intent", "study_agent", "textbook_flow", "services", "api", "eval"},
}


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


def test_cross_package_imports_follow_layer_direction() -> None:
    violations: list[str] = []
    for package, forbidden in FORBIDDEN.items():
        package_dir = APP_ROOT / package
        if not package_dir.is_dir():
            continue
        for path in package_dir.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for module in _imported_modules(path):
                target = module.split(".")[1]
                if target in forbidden:
                    violations.append(f"{path.relative_to(APP_ROOT.parent)}: {package} -> {module}")

    assert not violations, "跨层 import 违反依赖方向：\n" + "\n".join(violations)
