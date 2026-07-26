from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
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
_SAFE_PATTERN = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s]+$")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_VALID_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_TOKENS = {
    "__", "import", "lambda", "eval", "exec", "open", "globals", "locals",
    "compile", "subprocess", "os", "sys", "class", "def", "for", "while",
}


class FormulaError(ValueError):
    """사용자 이론식이 안전하지 않거나 계산 불가능할 때 발생한다."""


def validate_variable_names(variable_names: Sequence[str]) -> List[str]:
    names = [str(name).strip() for name in variable_names]
    if not names:
        raise FormulaError("독립변수를 하나 이상 선택하세요.")
    if len(set(names)) != len(names):
        raise FormulaError("독립변수 이름이 중복되어 있습니다.")

    reserved = set(_ALLOWED_FUNCTIONS) | set(_ALLOWED_CONSTANTS) | {"y_exp"}
    invalid = [
        name for name in names
        if not _VALID_NAME_PATTERN.fullmatch(name) or name in reserved
    ]
    if invalid:
        raise FormulaError(
            "독립변수 열 이름은 영문자 또는 밑줄로 시작하고 영문자·숫자·밑줄만 사용할 수 있습니다. "
            f"수정이 필요한 열: {', '.join(invalid)}"
        )
    return names


@dataclass(frozen=True)
class ParsedFormula:
    expression: sp.Expr
    variable_names: List[str]
    parameter_names: List[str]

    def build_function(
        self,
        parameter_values: Dict[str, float],
    ) -> Callable[[object], np.ndarray]:
        missing = [name for name in self.parameter_names if name not in parameter_values]
        if missing:
            raise FormulaError(f"다음 매개변수 값이 없습니다: {', '.join(missing)}")

        variable_symbols = [sp.Symbol(name, real=True) for name in self.variable_names]
        parameter_symbols = [sp.Symbol(name, real=True) for name in self.parameter_names]
        numeric = sp.lambdify(
            [*variable_symbols, *parameter_symbols],
            self.expression,
            modules=["numpy"],
        )
        ordered_parameters = [float(parameter_values[name]) for name in self.parameter_names]

        def evaluate(input_values: object) -> np.ndarray:
            arrays = _coerce_variable_arrays(input_values, self.variable_names)
            n_samples = len(arrays[0])
            try:
                result = numeric(*arrays, *ordered_parameters)
            except Exception as exc:
                raise FormulaError(f"이론식을 계산하지 못했습니다: {exc}") from exc

            result_array = np.asarray(result, dtype=float)
            if result_array.ndim == 0:
                result_array = np.full(n_samples, float(result_array), dtype=float)
            else:
                try:
                    result_array = np.broadcast_to(result_array, (n_samples,)).astype(float)
                except ValueError as exc:
                    raise FormulaError(
                        "이론식의 출력 크기가 데이터 행 수와 맞지 않습니다."
                    ) from exc

            if not np.all(np.isfinite(result_array)):
                raise FormulaError(
                    "이론식 계산 결과에 NaN 또는 무한대가 포함되어 있습니다. "
                    "정의역과 매개변수 값을 확인하세요."
                )
            return result_array

        return evaluate


def _coerce_variable_arrays(input_values: object, variable_names: Sequence[str]) -> List[np.ndarray]:
    names = list(variable_names)

    if isinstance(input_values, pd.DataFrame):
        missing = [name for name in names if name not in input_values.columns]
        if missing:
            raise FormulaError(f"이론식 계산에 필요한 열이 없습니다: {', '.join(missing)}")
        return [input_values[name].to_numpy(dtype=float) for name in names]

    if isinstance(input_values, Mapping):
        missing = [name for name in names if name not in input_values]
        if missing:
            raise FormulaError(f"이론식 계산에 필요한 변수가 없습니다: {', '.join(missing)}")
        raw_arrays = [np.asarray(input_values[name], dtype=float) for name in names]
        try:
            broadcast = np.broadcast_arrays(*raw_arrays)
        except ValueError as exc:
            raise FormulaError("독립변수 배열의 크기가 서로 맞지 않습니다.") from exc
        return [np.asarray(array, dtype=float).ravel() for array in broadcast]

    array = np.asarray(input_values, dtype=float)
    n_features = len(names)
    if array.ndim == 0:
        if n_features != 1:
            raise FormulaError("여러 독립변수를 사용할 때는 각 변수의 값을 모두 입력해야 합니다.")
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if n_features == 1:
            array = array.reshape(-1, 1)
        elif array.size == n_features:
            array = array.reshape(1, -1)
        else:
            raise FormulaError(
                f"입력 배열의 열 수가 독립변수 수({n_features})와 맞지 않습니다."
            )
    elif array.ndim != 2:
        raise FormulaError("독립변수 입력은 1차원 또는 2차원 배열이어야 합니다.")

    if array.shape[1] != n_features:
        raise FormulaError(
            f"입력 배열의 열 수({array.shape[1]})가 독립변수 수({n_features})와 맞지 않습니다."
        )
    return [array[:, index] for index in range(n_features)]


def discover_parameter_names(
    formula_text: str,
    variable_names: Iterable[str] = ("x",),
) -> List[str]:
    if not formula_text or not formula_text.strip():
        return []
    variables = set(validate_variable_names(list(variable_names)))
    reserved = set(_ALLOWED_FUNCTIONS) | set(_ALLOWED_CONSTANTS) | variables
    names = set(_IDENTIFIER_PATTERN.findall(formula_text))
    return sorted(name for name in names if name not in reserved)


def parse_formula(
    formula_text: str,
    parameter_names: Iterable[str],
    variable_names: Iterable[str] = ("x",),
) -> ParsedFormula:
    text = (formula_text or "").strip()
    if not text:
        raise FormulaError("이론식을 입력하세요.")
    if len(text) > 500:
        raise FormulaError("이론식이 너무 깁니다. 500자 이하로 입력하세요.")
    if not _SAFE_PATTERN.fullmatch(text):
        raise FormulaError("이론식에는 숫자, 영문 변수, 산술기호, 괄호만 사용할 수 있습니다.")

    lowered = text.lower()
    if any(token in lowered for token in _FORBIDDEN_TOKENS):
        raise FormulaError("허용되지 않은 표현이 포함되어 있습니다.")

    variables = validate_variable_names(list(variable_names))
    variable_symbols = {name: sp.Symbol(name, real=True) for name in variables}
    reserved = set(_ALLOWED_FUNCTIONS) | set(_ALLOWED_CONSTANTS) | set(variables)

    params = sorted(set(str(name) for name in parameter_names))
    invalid_parameters = [
        name for name in params
        if not _VALID_NAME_PATTERN.fullmatch(name) or name in reserved
    ]
    if invalid_parameters:
        raise FormulaError(f"사용할 수 없는 매개변수 이름: {', '.join(invalid_parameters)}")

    local_dict = {
        **variable_symbols,
        **_ALLOWED_FUNCTIONS,
        **_ALLOWED_CONSTANTS,
        **{name: sp.Symbol(name, real=True) for name in params},
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
        expression = parse_expr(
            text,
            local_dict=local_dict,
            global_dict=global_dict,
            transformations=transformations,
            evaluate=True,
        )
    except Exception as exc:
        raise FormulaError(f"이론식을 해석하지 못했습니다: {exc}") from exc

    allowed_symbols = {
        *variable_symbols.values(),
        *[local_dict[name] for name in params],
    }
    unknown_symbols = expression.free_symbols - allowed_symbols
    if unknown_symbols:
        unknown = ", ".join(sorted(str(symbol) for symbol in unknown_symbols))
        raise FormulaError(f"정의되지 않은 변수가 있습니다: {unknown}")

    if not expression.free_symbols.intersection(variable_symbols.values()):
        raise FormulaError("이론식에는 선택한 독립변수가 하나 이상 포함되어야 합니다.")

    return ParsedFormula(
        expression=expression,
        variable_names=variables,
        parameter_names=params,
    )
