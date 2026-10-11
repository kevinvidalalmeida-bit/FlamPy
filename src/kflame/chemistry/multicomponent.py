"""Native multicomponent and Soret transport kernels.

The expensive face evaluation uses compiled Numba kernels and LAPACK;
it does *not* call Cantera while a residual, Jacobian, or flame is evaluated.
Collision-integral polynomials, viscosity and binary diffusion are fitted
natively from molecular parameters and the bundled Monchick--Mason tables.
Neither construction nor evaluation imports Cantera. The formulation and data
are attributed in CANTERA_TRANSPORT_LICENSE.txt (Cantera, BSD-3-Clause).

The implementation follows the Dixon--Lewis linear-system formulation used by
the multicomponent transport model: the ``L00`` block gives the ordinary
multicomponent diffusion matrix and the coupled 3N system gives the thermal
diffusion coefficients and conductivity.  It is intentionally kept in its own
module so the mature mixture-averaged fast path remains unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kflame.chemistry.mechanism import MechanismData, R_UNIV
from kflame.chemistry.transport import NativeTransport
from kflame.chemistry.collision_integrals import native_transport_fits
from kflame.chemistry.multicomponent_kernels import (
    thermal_system, evaluate_faces, temperature_terms,
    _temperature_terms_symmetric, _evaluate_faces_symmetric,
    _evaluate_faces_blocked,
)


_TINY_X = 1.0e-20
_MIN_C_INTERNAL = 1.0e-3


def _certifiable_temperature_inputs(inputs: tuple[np.ndarray, ...]) -> bool:
    """Check actual pair coefficient bits, never floating-point tolerances."""
    if len(inputs) != 10 or not all(type(x) is np.ndarray for x in inputs):
        return False
    n = inputs[0].size
    shapes = ((n,), (n, n), (n,), (n,), (n,), (n, 5),
              (n, n, 5), (n, n, 9), (n, n, 9), (n, n, 9))
    if n < 1 or not all(x.dtype == np.dtype(np.float64) and x.dtype.isnative
                       and x.flags.c_contiguous and x.flags.aligned
                       and x.shape == shape and np.isfinite(x).all()
                       for x, shape in zip(inputs, shapes)):
        return False
    if np.any(inputs[0] <= 0.0) or np.any(inputs[2] <= 0.0):
        return False
    return all(np.array_equal(inputs[i].view(np.uint64),
                              inputs[i].view(np.uint64).swapaxes(0, 1))
               for i in (1, 6, 7, 8, 9))


def _separate_temperature_outputs(inputs, outputs) -> bool:
    n = inputs[0].size
    shapes = ((n,), (n,), (n, n), (n, n), (n, n), (n, n))
    if len(outputs) != 6 or not all(type(x) is np.ndarray
            and x.dtype == np.dtype(np.float64) and x.shape == shape
            and x.flags.c_contiguous and x.flags.aligned and x.flags.writeable
            for x, shape in zip(outputs, shapes)):
        return False
    return all(not np.shares_memory(out, other)
               for i, out in enumerate(outputs) for other in (*inputs, *outputs[:i]))


@dataclass(frozen=True, init=False, eq=False)
class TemperatureMechanismSnapshot:
    """Owned immutable temperature inputs, certified once for their lifetime.

    Explicit construction opts into isolation from subsequent caller mutation.
    Bytes-backed arrays cannot have WRITEABLE re-enabled, even through .base.
    Raw-array Numba APIs always retain their original ordered-pair behavior.
    """

    _inputs: tuple[np.ndarray, ...]
    certified: bool

    def __init__(self, inputs: tuple[np.ndarray, ...]) -> None:
        if len(inputs) != 10:
            raise ValueError("Expected ten temperature input arrays")
        # Remember unsupported layouts before normalization; check values only
        # after copying so certification never belongs to caller-owned storage.
        layout_ok = all(type(x) is np.ndarray and x.flags.c_contiguous
                        and x.flags.aligned for x in inputs)
        copies = tuple(np.array(x, copy=True, order="C") for x in inputs)
        frozen = tuple(np.frombuffer(x.tobytes(), dtype=x.dtype).reshape(x.shape)
                       for x in copies)
        object.__setattr__(self, "_inputs", frozen)
        object.__setattr__(self, "certified",
                           layout_ok and _certifiable_temperature_inputs(frozen))

    @property
    def inputs(self) -> tuple[np.ndarray, ...]:
        # Expose views with independent ndarray metadata. Changing a returned
        # view's shape/dtype/strides cannot invalidate the private certificate.
        return tuple(x.view() for x in self._inputs)

    def _matches_bindings(self, current, bound) -> bool:
        return all(x is saved and x.dtype == owned.dtype and x.shape == owned.shape
                   and x.strides == owned.strides and not x.flags.writeable
                   and x.flags.aligned
                   for x, saved, owned in zip(current, bound, self._inputs))

    def evaluate(self, temperature: float, outputs: tuple[np.ndarray, ...]) -> bool:
        """Fill six outputs; return whether the certified route was selected.

        Unusual/overlapping buffers take the original route, including its
        existing exceptions. No buffer copying or domain clamping is added.
        This Python boundary checks buffers; compiled face loops allocate their
        own independent buffers and bypass this per-call Python check.
        """
        symmetric = (self.certified and math.isfinite(temperature) and temperature > 0.0
                     and _separate_temperature_outputs(self._inputs, outputs))
        kernel = _temperature_terms_symmetric if symmetric else temperature_terms
        kernel(temperature, *self._inputs, *outputs)
        return symmetric

    def evaluate_faces(self, temperature, pressure, y, cp_r, inv_w, crot, dense):
        mw, log_eps, epsilon, zrot, frot, visc, diff, astar, bstar, cstar = self._inputs
        if mw.size >= 24 and temperature.size >= 32 and not dense:
            return _evaluate_faces_blocked(
                temperature, pressure, y, cp_r, mw, inv_w, log_eps,
                epsilon, crot, zrot, frot, visc, diff, astar, bstar, cstar,
                dense, self.certified, 4)
        kernel = _evaluate_faces_symmetric if self.certified else evaluate_faces
        return kernel(temperature, pressure, y, cp_r, mw, inv_w, log_eps,
                      epsilon, crot, zrot, frot, visc, diff, astar, bstar, cstar, dense)


def _frot(reduced_temperature: np.ndarray) -> np.ndarray:
    """Parker correction for the rotational collision number."""
    tr = np.maximum(np.asarray(reduced_temperature, dtype=float), 1.0e-300)
    sqrt_tr = np.sqrt(tr)
    return (
        1.0
        + 0.5 * math.sqrt(math.pi) * math.pi * sqrt_tr
        + (0.25 * math.pi * math.pi + 2.0) * tr
        + math.sqrt(math.pi) * math.pi * sqrt_tr * tr
    )


def _poly8(z: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate Cantera's ascending-order eighth-degree polynomial table."""
    result = np.asarray(coefficients[..., 8], dtype=float).copy()
    for degree in range(7, -1, -1):
        result = result * z + coefficients[..., degree]
    return result


class NativeMulticomponentTransport:
    """Multicomponent/Soret closure evaluated without per-face Cantera calls.

    Parameters
    ----------
    mech:
        Mechanism data used by the native thermo/kinetics kernels.
    mechanism_path:
        Retained for caller compatibility; all data come from ``mech``.
    base_transport:
        Existing native transport object, supplying pair molecular parameters.
        Binary-diffusion and viscosity fits are constructed natively for this
        mechanism; no exported Cantera fits are used by this closure.
    """

    def __init__(
        self,
        mech: MechanismData,
        mechanism_path: str | Path,
        base_transport: NativeTransport | None = None,
        linear_solver: str = "schur",
    ) -> None:
        self.mech = mech
        self.n_species = int(mech.n_species)
        self.mw = np.asarray(mech.molecular_weights, dtype=float)
        self.inv_w = np.asarray(mech.inv_molecular_weights, dtype=float)
        self.epsilon_k = np.maximum(np.asarray(mech.well_depth, dtype=float), 1.0e-300)
        geometry = np.asarray(mech.geometry, dtype=int)
        self.crot = np.where(geometry == 0, 0.0, np.where(geometry == 1, 1.0, 1.5))
        self.zrot = np.asarray(mech.rot_relax, dtype=float)
        self.frot_298 = _frot(self.epsilon_k / 298.0)
        self.base_transport = base_transport or NativeTransport(mech)
        if linear_solver not in ("schur", "dense"):
            raise ValueError("Unknown transport linear solver")
        self.linear_solver = linear_solver
        self.use_compiled_faces = True
        stars, self._viscosity_fit, self._diffusion_fit = native_transport_fits(mech, self.base_transport)
        self._astar_poly, self._bstar_poly, self._cstar_poly = stars
        self._log_eps_pair = np.log(np.maximum(self.base_transport._eps_pair, 1.0e-300))
        self._temperature_snapshot = TemperatureMechanismSnapshot(self._temperature_inputs())
        self._temperature_snapshot_bindings = self._temperature_snapshot.inputs
        (self.mw, self._log_eps_pair, self.epsilon_k, self.zrot, self.frot_298,
         self._viscosity_fit, self._diffusion_fit, self._astar_poly,
         self._bstar_poly, self._cstar_poly) = self._temperature_snapshot_bindings

    def _temperature_inputs(self) -> tuple[np.ndarray, ...]:
        return (self.mw, self._log_eps_pair, self.epsilon_k, self.zrot, self.frot_298,
                self._viscosity_fit, self._diffusion_fit, self._astar_poly,
                self._bstar_poly, self._cstar_poly)

    def export_temperature_snapshot(self) -> tuple[tuple[np.ndarray, ...], bool]:
        """Export immutable kernel arrays and their snapshot-bound certificate.

        Order: mw, log_eps_pair, epsilon, zrot, frot298, viscosity_fit,
        diffusion_fit, astar_fit, bstar_fit, cstar_fit. Keep this complete set
        together: the flag does not certify caller arrays or later replacements.
        A compiled consumer must also establish independent writable outputs.

        Rebound/metadata-edited transport arrays export their current values as
        new immutable copies with certified=False, even when still symmetric.
        This preserves the original route for arbitrary user modifications.
        """
        current = self._temperature_inputs()
        snapshot = self._temperature_snapshot
        if snapshot._matches_bindings(current, self._temperature_snapshot_bindings):
            return snapshot.inputs, snapshot.certified
        return TemperatureMechanismSnapshot(current).inputs, False

    def _viscosities(self, temperature: float) -> np.ndarray:
        values = np.polynomial.polynomial.polyval(math.log(temperature), self._viscosity_fit.T)
        return math.sqrt(temperature) * values * values

    def _temperature_data(
        self, temperature: float, cp_r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``bdiff, A*, B*, C*, Zrot(T)`` for one face."""
        temperature = float(temperature)
        # The pair well depth includes the polar correction.  It must be the
        # same one used by the binary-diffusion fit; using only the geometric
        # mean is harmless for non-polar pairs but biases H2O-containing states.
        z_pair = math.log(temperature) - self._log_eps_pair
        astar = _poly8(z_pair, self._astar_poly)
        bstar = _poly8(z_pair, self._bstar_poly)
        cstar = _poly8(z_pair, self._cstar_poly)

        bdiff = temperature ** 1.5 * np.polynomial.polynomial.polyval(
            math.log(temperature), self._diffusion_fit.transpose(2, 0, 1))
        viscosity = self._viscosities(temperature)
        diagonal = 1.2 * R_UNIV * temperature * viscosity * np.diag(astar) / self.mw
        np.fill_diagonal(bdiff, diagonal)

        rot_relax = np.maximum(1.0, self.zrot) * self.frot_298 / _frot(
            self.epsilon_k / temperature
        )
        return bdiff, astar, bstar, cstar, rot_relax

    def _composition(self, mass_fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        y = np.asarray(mass_fractions, dtype=float)
        if y.shape != (self.n_species,):
            raise ValueError("La composici\u00f3n de cara tiene una forma incompatible.")
        inv_wmix = float(np.dot(y, self.inv_w))
        wmix = 1.0 / max(inv_wmix, 1.0e-300)
        x = np.maximum(y * wmix * self.inv_w, _TINY_X)
        return y, x, wmix

    def _l00(
        self, temperature: float, x: np.ndarray, bdiff: np.ndarray
    ) -> np.ndarray:
        """Build the L00 block of the multicomponent transport system."""
        inv_bdiff = 1.0 / np.maximum(bdiff, 1.0e-300)
        summed = inv_bdiff @ x - x * np.diag(inv_bdiff)
        summed /= self.mw
        l00 = (16.0 * float(temperature) / 25.0) * x[None, :] * (
            self.mw[None, :] * summed[:, None] + x[:, None] * inv_bdiff
        )
        np.fill_diagonal(l00, 0.0)
        return l00

    def _multi_diffusion(
        self, temperature: float, pressure: float, x: np.ndarray, wmix: float, bdiff: np.ndarray,
        inverse: np.ndarray | None = None,
    ) -> np.ndarray:
        if inverse is None:
            inverse = np.linalg.solve(self._l00(temperature, x, bdiff), np.eye(self.n_species))
        prefactor = 16.0 * float(temperature) * wmix / (25.0 * float(pressure))
        return (
            prefactor
            * (x[:, None] / self.mw[None, :])
            * (inverse - np.diag(inverse)[:, None])
        )

    def _thermal_system(
        self,
        temperature: float,
        x: np.ndarray,
        cp_r: np.ndarray,
        bdiff: np.ndarray,
        astar: np.ndarray,
        bstar: np.ndarray,
        cstar: np.ndarray,
        rot_relax: np.ndarray,
        l00_inverse: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Solve the coupled 3N Dixon--Lewis system for Soret coefficients."""
        return thermal_system(
            temperature, x, cp_r, bdiff, astar, bstar, cstar, rot_relax,
            self.mw, self.crot, self._viscosities(temperature),
            self._l00(temperature, x, bdiff), l00_inverse,
            self.linear_solver == "dense",
        )

    def eval_face(
        self, temperature: float, pressure: float, mass_fractions: np.ndarray, cp_r: np.ndarray
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        """Evaluate ``rho, lambda, Wmix, D_multi, D_thermal`` at one face."""
        y, x, wmix = self._composition(mass_fractions)
        bdiff, astar, bstar, cstar, rot_relax = self._temperature_data(temperature, cp_r)
        inverse = np.linalg.solve(self._l00(temperature, x, bdiff), np.eye(self.n_species))
        multi = self._multi_diffusion(temperature, pressure, x, wmix, bdiff, inverse=inverse)
        dthermal, conductivity = self._thermal_system(
            temperature, x, cp_r, bdiff, astar, bstar, cstar, rot_relax,
            l00_inverse=inverse,
        )
        rho = float(pressure) * wmix / (R_UNIV * float(temperature))
        return rho, conductivity, wmix, multi, dthermal

    def eval_faces(
        self,
        temperature: np.ndarray,
        pressure: float,
        mass_fractions: np.ndarray,
        cp_r: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Independent face systems, with exact Schur reduction by default."""
        temperature = np.asarray(temperature, dtype=float)
        mass_fractions = np.asarray(mass_fractions, dtype=float)
        cp_r = np.asarray(cp_r, dtype=float)
        if mass_fractions.shape != (self.n_species, temperature.size):
            raise ValueError("Y_face debe tener forma (n_species, n_faces).")
        if cp_r.shape != mass_fractions.shape:
            raise ValueError("cp_R debe tener la misma forma que Y_face.")

        if self.use_compiled_faces:
            snapshot = self._temperature_snapshot
            if snapshot._matches_bindings(self._temperature_inputs(),
                                          self._temperature_snapshot_bindings):
                return snapshot.evaluate_faces(
                    temperature, float(pressure), mass_fractions, cp_r,
                    self.inv_w, self.crot, self.linear_solver == "dense",
                )
            return evaluate_faces(
                temperature, float(pressure), mass_fractions, cp_r, self.mw,
                self.inv_w, self._log_eps_pair, self.epsilon_k, self.crot,
                self.zrot, self.frot_298, self._viscosity_fit, self._diffusion_fit,
                self._astar_poly, self._bstar_poly, self._cstar_poly,
                self.linear_solver == "dense",
            )

        faces = temperature.size
        rho = np.empty(faces, dtype=float)
        conductivity = np.empty(faces, dtype=float)
        wmix = np.empty(faces, dtype=float)
        multi = np.empty((self.n_species, self.n_species, faces), dtype=float)
        dthermal = np.empty((self.n_species, faces), dtype=float)
        for face in range(faces):
            rho[face], conductivity[face], wmix[face], multi[:, :, face], dthermal[:, face] = (
                self.eval_face(
                    temperature[face], pressure, mass_fractions[:, face], cp_r[:, face]
                )
            )
        return rho, conductivity, wmix, multi, dthermal
