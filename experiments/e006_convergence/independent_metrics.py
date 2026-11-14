"""Recompute checkpoint validation and reference errors with independent spatial autograd."""
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e005_evolution_suite"))
from accept_results import constraints
from audit_physics import mlp, pde_mse
import longrun as old


class IndependentMetrics:
    def __init__(self, case, seed):
        self.case = case
        self.grid, self.truth = old.eval_grid(case)
        self.validation = old.Problem(case, 70000 + seed, 192, 32)
        self.audit = old.Problem(case, 991, 384, 48)

    def losses(self, p, theta):
        return [pde_mse(self.case, theta, p.x, p.lo, p.hi, p.nout), *constraints(p, theta)]

    def relative_error(self, theta):
        p = self.validation
        with torch.no_grad():
            y = torch.cat([mlp(theta, x, p.lo, p.hi, p.nout) for x in self.grid.split(4096)])
        return float((y-self.truth).norm()/self.truth.norm())
