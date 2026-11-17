"""A safe arithmetic evaluator for the agent's `calculator` tool.

Never `eval()` model output. We parse it with `ast` and walk an allow-listed subset:
numbers, + - * / // % **, unary +/-, parentheses, and a few pure functions.
"""
from __future__ import annotations

import ast
import math
import re
import operator

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt, "sum": lambda *a: sum(a)}


class CalcError(ValueError):
    pass


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise CalcError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise CalcError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> float:
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression)  # 896,000 -> 896000 but keep f(a, b)
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"not a valid arithmetic expression: {expression!r}") from e
    try:
        result = _eval(tree)
    except ZeroDivisionError as e:
        raise CalcError("division by zero") from e
    if isinstance(result, float):
        result = round(result, 6)
    return result
