"""A calculator the model can call instead of doing arithmetic in its head.

LLMs are unreliable at "what is (5,082 - 4,941) / 4,941"; filings questions are
full of growth rates and ratios. The expression is parsed with `ast` and only
numbers and + - * / ** % ( ) are allowed, so there is no eval() of model output.
"""

from __future__ import annotations

import ast
import operator

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalcError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise CalcError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise CalcError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> float:
    cleaned = expression.replace(",", "").replace("$", "").replace("_", "").strip()
    if len(cleaned) > 200:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"not an arithmetic expression: {expression!r}") from exc
    try:
        return round(_eval(tree), 6)
    except ZeroDivisionError as exc:
        raise CalcError("division by zero") from exc
