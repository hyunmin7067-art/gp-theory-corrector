from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_ALLOWED_FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
}
_ALLOWED_CONSTANTS = {"pi": sp.pi, "e": sp.E}
_RESERVED_NAMES = set(_ALLOWED_FUNCTIONS) | set(_ALLOWED_CONSTANTS) | {"x"}
_SAFE_PATTERN = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s]+$")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FORBIDDEN_TOKENS = {
    "__", "import", "lambda", "eval", "exec", "open", "globals", "locals",
    "compile", "subprocess", "os", "sys", "class", "def", "for", "while",
}


class FormulaError(ValueError):
    """사용자 이론식이 안전하지 않거나 계산 불가능할 때 발생한다."""


@dataclass(frozen=True)
class ParsedFormula:
    expression: sp.Expr
    parameter_names: List[str]

    def build_function(self, parameter_values: Dict[str, float]) -> Callable[[np.ndarray], np.ndarray]:
        missing = [name for name in self.parameter_names if name not in parameter_values]
        if missing:
            raise FormulaError(f"다음 매개변수 값이 없습니다: {', '.join(missing)}")

        x_symbol = sp.Symbol("x", real=True)
        param_symbols = [sp.Symbol(name, real=True) for name in self.parameter_names]
        numeric = sp.lambdify([x_symbol, *param_symbols], self.expression, modules=["numpy"])
        ordered_values = [float(parameter_values[name]) for name in self.parameter_names]

        def evaluate(x_values: np.ndarray) -> np.ndarray:
            x_array = np.asarray(x_values, dtype=float)
            try:
                result = numeric(x_array, *ordered_values)
            except Exception as exc:
                raise FormulaError(f"이론식을 계산하지 못했습니다: {exc}") from exc

            result_array = np.asarray(result, dtype=float)
            if result_array.ndim == 0:
                result_array = np.full_like(x_array, float(result_array), dtype=float)
            else:
                result_array = np.broadcast_to(result_array, x_array.shape).astype(float)

            if not np.all(np.isfinite(result_array)):
                raise FormulaError(
                    "이론식 계산 결과에 NaN 또는 무한대가 포함되어 있습니다. "
                    "정의역과 매개변수 값을 확인하세요."
                )
            return result_array

        return evaluate


def discover_parameter_names(formula_text: str) -> List[str]:
    if not formula_text or not formula_text.strip():
        return []
    names = set(_IDENTIFIER_PATTERN.findall(formula_text))
    return sorted(name for name in names if name not in _RESERVED_NAMES)


def parse_formula(formula_text: str, parameter_names: Iterable[str]) -> ParsedFormula:
    text = (formula_text or "").strip()
    if not text:
        raise FormulaError("이론식을 입력하세요.")
    if len(text) > 300:
        raise FormulaError("이론식이 너무 깁니다. 300자 이하로 입력하세요.")
    if not _SAFE_PATTERN.fullmatch(text):
        raise FormulaError("이론식에는 숫자, 영문 변수, 산술기호, 괄호만 사용할 수 있습니다.")
    lowered = text.lower()
    if any(token in lowered for token in _FORBIDDEN_TOKENS):
        raise FormulaError("허용되지 않은 표현이 포함되어 있습니다.")

    param_names = sorted(set(parameter_names))
    invalid_names = [
        name for name in param_names
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        or name in _RESERVED_NAMES
    ]
    if invalid_names:
        raise FormulaError(f"사용할 수 없는 매개변수 이름: {', '.join(invalid_names)}")

    x_symbol = sp.Symbol("x", real=True)
    local_dict = {
        "x": x_symbol,
        **_ALLOWED_FUNCTIONS,
        **_ALLOWED_CONSTANTS,
        **{name: sp.Symbol(name, real=True) for name in param_names},
    }
    global_dict = {
        "__builtins__": {},
        "Integer": sp.Integer,
        "Float": sp.Float,
        "Rational": sp.Rational,
        "Symbol": sp.Symbol,
    }
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    try:
        expr = parse_expr(
            text,
            local_dict=local_dict,
            global_dict=global_dict,
            transformations=transformations,
            evaluate=True,
        )
    except Exception as exc:
        raise FormulaError(f"이론식을 해석하지 못했습니다: {exc}") from exc

    allowed_symbols = {x_symbol, *[local_dict[name] for name in param_names]}
    unknown_symbols = expr.free_symbols - allowed_symbols
    if unknown_symbols:
        unknown = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
        raise FormulaError(f"정의되지 않은 변수가 있습니다: {unknown}")
    if not expr.has(x_symbol):
        raise FormulaError("이론식에는 독립변수 x가 포함되어야 합니다.")

    return ParsedFormula(expression=expr, parameter_names=param_names)
