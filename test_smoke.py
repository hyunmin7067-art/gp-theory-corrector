import numpy as np

from formula_utils import discover_parameter_names, parse_formula
from gp_core import analyze_kernels
from sample_data import make_rotation_sample


def main():
    data = make_rotation_sample()
    formula = "2*pi*sqrt(x/kappa)"
    params = discover_parameter_names(formula)
    parsed = parse_formula(formula, params)
    fn = parsed.build_function({"kappa": 0.08})

    x = data["x"].to_numpy()
    y = data["y_exp"].to_numpy()
    theory = fn(x)
    result = analyze_kernels(x, y, theory, restarts=0)

    assert len(result.metrics) == 4
    assert result.best_kernel_name in result.fitted_models
    mean, std = result.fitted_models[result.best_kernel_name].predict(x)
    assert mean.shape == x.shape
    assert std.shape == x.shape
    assert np.all(std > 0)
    print(result.metrics[["Kernel", "CV_RMSE", "NLPD"]])
    print("Best:", result.best_kernel_name)


if __name__ == "__main__":
    main()
