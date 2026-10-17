"""Convenience constructor for JointInputs.

This is the IO boundary: the one place where values arrive in human units and
leave in SI. Everything downstream is SI. Keyword names carry their unit
(`sigma_a_um`, `pressure_MPa`) so a caller cannot pass the right number in the
wrong unit without it being visible at the call site.
"""

from __future__ import annotations

from .schema import (
    FlagSet,
    JointInputs,
    Provenance,
    Quantity,
    SurfaceSpec,
    load_gas,
    load_material,
)


def build_joint(
    material_a: str,
    material_b: str,
    *,
    sigma_a_um: float,
    sigma_b_um: float,
    pressure_MPa: float,
    temperature_K: float,
    slope_a: float | None = None,
    slope_b: float | None = None,
    gas: str | None = None,
    gas_pressure_kPa: float | None = None,
    accommodation_a: float | None = None,
    accommodation_b: float | None = None,
    regime_override: str | None = None,
    # relative (1-sigma) uncertainties; None means "unquantified, held fixed"
    sigma_rel_unc: float | None = 0.10,
    slope_rel_unc: float | None = 0.20,
    pressure_rel_unc: float | None = 0.05,
    hardness_rel_unc: float | None = 0.30,
    label: str = "",
    flags: FlagSet | None = None,
) -> JointInputs:
    """Assemble a JointInputs from values in laboratory units.

    `gas=None` means vacuum. Default relative uncertainties reflect typical
    experimental scatter: profilometry roughness to ~10 %, slope to ~20 %,
    load cell to ~5 %, and microhardness to ~30 % because it varies with
    indentation load and surface preparation. Pass None for any of them to
    hold that quantity fixed instead.
    """
    flags = flags or FlagSet()
    a = load_material(material_a, flags)
    b = load_material(material_b, flags)

    if hardness_rel_unc is not None:
        a.Hv = a.Hv.with_(uncertainty=a.Hv.si * hardness_rel_unc)
        b.Hv = b.Hv.with_(uncertainty=b.Hv.si * hardness_rel_unc)

    def surf(sigma_um: float, slope: float | None) -> SurfaceSpec:
        s = Quantity.parse(
            sigma_um, "um", rel_uncertainty=sigma_rel_unc,
            provenance=Provenance.USER_MEASURED, name="sigma",
        )
        m = None
        if slope is not None:
            m = Quantity.parse(
                slope, "-", rel_uncertainty=slope_rel_unc,
                provenance=Provenance.USER_MEASURED, name="m",
            )
        return SurfaceSpec(sigma=s, m=m)

    gas_spec = None
    if gas is not None:
        if gas_pressure_kPa is None:
            raise ValueError(
                "gas_pressure_kPa is required when a gas is specified; the "
                "mean free path, and hence the rarefaction correction, depends "
                "on it directly"
            )
        gas_spec = load_gas(gas, gas_pressure_kPa * 1e3, flags)

    def acc(v: float | None) -> Quantity | None:
        if v is None:
            return None
        return Quantity.parse(
            v, "-", provenance=Provenance.USER_MEASURED, name="accommodation"
        )

    return JointInputs(
        material_a=a,
        material_b=b,
        surface_a=surf(sigma_a_um, slope_a),
        surface_b=surf(sigma_b_um, slope_b),
        pressure=Quantity.parse(
            pressure_MPa, "MPa", rel_uncertainty=pressure_rel_unc,
            provenance=Provenance.USER_MEASURED, name="P",
        ),
        temperature=Quantity.parse(
            temperature_K, "K", provenance=Provenance.USER_MEASURED, name="T"
        ),
        gas=gas_spec,
        accommodation_a=acc(accommodation_a),
        accommodation_b=acc(accommodation_b),
        regime_override=regime_override,
        label=label or f"{material_a}/{material_b}",
    )
