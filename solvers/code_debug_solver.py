from __future__ import annotations

import ast
import copy
import re

from solvers.common import LocalAnswer


def solve_code_debug(prompt: str) -> LocalAnswer | None:
    code = _extract_code(prompt)
    if not code:
        return None

    extremum = _repair_extremum_function(prompt, code)
    if extremum is not None:
        return LocalAnswer(extremum, 0.98, "extremum_repair")

    mutable_default = _repair_mutable_default(prompt, code)
    if mutable_default is not None:
        return LocalAnswer(mutable_default, 0.99, "mutable_default_repair")

    index = _repair_len_index(prompt, code)
    if index is not None:
        return LocalAnswer(index, 0.99, "len_index_repair")

    try:
        ast.parse(code)
    except SyntaxError:
        repaired = _repair_missing_colons(code)
        if repaired != code:
            try:
                ast.parse(repaired)
            except SyntaxError:
                return None
            return LocalAnswer(repaired, 0.97, "missing_colon_repair")
        return None
    return None


def _extract_code(prompt: str) -> str | None:
    fenced = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", prompt, flags=re.S)
    if fenced:
        return fenced.group(1).strip()

    inline_function = re.search(
        r"\b(def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*:?.*?)"
        r"(?=\.\s*(?:Find|Fix|Identify|Explain|What|Why|Correct)\b|$)",
        prompt,
        flags=re.IGNORECASE | re.S,
    )
    if inline_function:
        return inline_function.group(1).strip().rstrip(".")
    return None


def _repair_extremum_function(prompt: str, code: str) -> str | None:
    intent_match = re.search(
        r"\b(?:return|find|compute|get)\s+(?:the\s+)?(max(?:imum)?|largest|min(?:imum)?|smallest)\b",
        prompt,
        flags=re.IGNORECASE,
    )
    function_match = re.search(
        r"\bdef\s+([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*:",
        code,
    )
    if not intent_match or not function_match:
        return None

    function_name, parameter = function_match.groups()
    target = intent_match.group(1).lower()
    builtin = "min" if target in {"min", "minimum", "smallest"} else "max"

    # Only claim a free repair when the implementation clearly returns one fixed element.
    fixed_element = re.search(
        fr"\breturn\s+{re.escape(parameter)}\s*\[\s*-?\d+\s*\]",
        code,
    )
    if not fixed_element:
        return None
    return f"def {function_name}({parameter}):\n    return {builtin}({parameter})"


def _repair_mutable_default(prompt: str, code: str) -> str | None:
    explicit_mutable_default = re.search(
        r"\bmutable[- ]default\b|\bmutable\b.{0,30}\bdefault\b",
        prompt,
        re.I,
    )
    persistent_state = re.search(
        r"\b(?:persist|retain|shared|remember)\b.{0,60}\b(?:across|between)\b.{0,20}\b(?:calls?|invocations?)\b",
        prompt,
        re.I,
    )
    if not explicit_mutable_default and not persistent_state:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) != 1:
        return None
    function = functions[0]
    positional = [*function.args.posonlyargs, *function.args.args]
    default_args = positional[len(positional) - len(function.args.defaults) :]
    repairs: list[tuple[str, ast.expr]] = []
    for argument, default in zip(default_args, function.args.defaults):
        factory = _mutable_factory(default)
        if factory is None:
            continue
        repairs.append((argument.arg, factory))
    if not repairs:
        return None

    for index, (argument, _) in enumerate(zip(default_args, function.args.defaults)):
        if any(argument.arg == name for name, _ in repairs):
            function.args.defaults[index] = ast.Constant(value=None)
    guards: list[ast.stmt] = []
    for name, factory in repairs:
        guards.append(
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id=name, ctx=ast.Load()),
                    ops=[ast.Is()],
                    comparators=[ast.Constant(value=None)],
                ),
                body=[
                    ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=factory,
                    )
                ],
                orelse=[],
            )
        )
    function.body = guards + function.body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _mutable_factory(default: ast.expr) -> ast.expr | None:
    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
        return copy.deepcopy(default)
    if (
        isinstance(default, ast.Call)
        and isinstance(default.func, ast.Name)
        and default.func.id in {"list", "dict", "set"}
        and not default.args
        and not default.keywords
    ):
        return copy.deepcopy(default)
    return None


def _repair_len_index(prompt: str, code: str) -> str | None:
    if not re.search(r"\bindexerror\b|\b(?:return|get|find)\s+(?:the\s+)?last\b", prompt, re.I):
        return None
    pattern = re.compile(
        r"(?P<container>\b[A-Za-z_]\w*)\s*\[\s*len\s*\(\s*(?P=container)\s*\)\s*\]"
    )
    repaired, count = pattern.subn(r"\g<container>[-1]", code)
    if count != 1:
        return None
    try:
        ast.parse(repaired)
    except SyntaxError:
        return None
    return repaired


def _repair_missing_colons(code: str) -> str:
    repaired_lines: list[str] = []
    header = re.compile(r"^\s*(?:async\s+def|def|class|if|elif|else|for|while|try|except|finally|with)\b")
    for line in code.splitlines():
        stripped = line.rstrip()
        if header.search(stripped) and not stripped.endswith(":"):
            stripped += ":"
        repaired_lines.append(stripped)
    return "\n".join(repaired_lines)
