"""FEM Toolbox: vectorized arbitrary-order CG/hp-DG finite elements on triangles."""

from femtoolbox.core.adapt import adaptive_solve, convergence_study, mark_dorfler, refine_h_selected, refine_h_uniform
from femtoolbox.core.advisor import recommend
from femtoolbox.core.domain import (AnnulusDomain, DiskDomain, LShapeDomain, ParametricDomain, PlateWithHoleDomain,
                                    PolygonDomain, RectangleDomain, SquareDomain)
from femtoolbox.core.elasticity import Elasticity, solve_elasticity
from femtoolbox.core.estimate import error_norms, residual_estimator, smoothness_indicator, zz_estimator
from femtoolbox.core.export import export_all_csv, export_mesh_csv, export_solution_csv, export_solution_vtu, export_vtu
from femtoolbox.core.meshio import read_msh, write_msh
from femtoolbox.core.mesh import Mesh2D, bisect, refine_uniform
from femtoolbox.core.pde import PDE, BoundaryCondition, parse_boundary_conditions
from femtoolbox.core.postprocess import boundary_flux, boundary_integral, integrate, quantities
from femtoolbox.core.solution import EigenSolution, FEMSolution, TransientFEMSolution
from femtoolbox.core.solver import solve, solve_eigen, solve_heat_theta, solve_transient
from femtoolbox.core.space import FunctionSpace

__version__ = "2.1.0"
__all__ = [n for n in dir() if not n.startswith("_")]
