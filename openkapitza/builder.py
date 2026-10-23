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
    fill=None,
    z_trunc: float | str | None = None,
    z_trunc_unc: float | None = None,
) -> JointInputs:
    """Assemble a JointInputs from values in laboratory units.

    `gas=None` means vacuum. Default relative uncertainties reflect typical
    experimental scatter: profilometry roughness to ~10 %, slope to ~20 %,
    load cell to ~5 %, and microhardness to ~30 % because it varies with
    indentation load and surface preparation. Pass None for any of them to
    hold that quantity fixed instead.

    `z_trunc` switches the plastic branch to the truncated-Gaussian light-load
    model (open issue P0-7): a number is the truncation level in units of
    sigma, fitted to light-load data on your surfaces; "reference" takes the
    published bead-blasted SS 304 / Ni 200 value at this sigma/m (needs both
    slopes; it was fitted with THEIR c1/c2, so it can over-correct with a
    different hardness fit -- see contact.truncated). `z_trunc_unc` is its
    1-sigma, propagated in the Monte Carlo.
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

    z_q = None
    if z_trunc == "reference":
        if slope_a is None or slope_b is None:
            raise ValueError('z_trunc="reference" needs slope_a and slope_b: the '
                             "published truncation levels are tabulated against sigma/m")
        from .contact.truncated import reference_truncation
        som = (sigma_a_um ** 2 + sigma_b_um ** 2) ** 0.5 / (slope_a ** 2 + slope_b ** 2) ** 0.5
        look = reference_truncation(material_a, som, flags) or reference_truncation(
            material_b, som, flags)
        if look is None:
            raise ValueError(f"No published truncation level for {material_a} or "
                             f"{material_b} (only bead-blasted SS304, Ni200); pass a "
                             "number fitted to your own light-load data")
        z_q = look.z_trunc
        if z_trunc_unc is not None:
            z_q = z_q.with_(uncertainty=z_trunc_unc)
    elif z_trunc is not None:
        z_q = Quantity.parse(float(z_trunc), "-", uncertainty=z_trunc_unc,
                             provenance=Provenance.USER_MEASURED, name="z_trunc")

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
        fill=fill,
        z_trunc=z_q,
        label=label or f"{material_a}/{material_b}",
    )


def tim_fill(blt_um: float, k_W_mK: float, *, tim_class: str | None = None,
             R_c_mm2K_W: float | None = None, R_measured_mm2K_W: float | None = None,
             label: str = "TIM", rel_unc: float | None = 0.10):
    """A filled joint in lab units: BLT (um), k (W/mK), boundaries/total in mm^2K/W."""
    from .network.stack import FillSpec, Layer
    q = lambda v, u, n: Quantity.parse(v, u, rel_uncertainty=rel_unc,
                                       provenance=Provenance.USER_MEASURED, name=n)
    return FillSpec(
        kind="tim",
        layers=[Layer(label, q(blt_um, "um", "BLT"), q(k_W_mK, "W/mK", "k_TIM"))],
        tim_class=tim_class,
        R_c_total=None if R_c_mm2K_W is None else q(R_c_mm2K_W * 1e-6, "m2K/W", "R_c"),
        R_measured=None if R_measured_mm2K_W is None else q(R_measured_mm2K_W * 1e-6, "m2K/W", "R_TIM"),
    )


def bond_fill(layers: list[tuple], *, h_K_a_MW: float | None = None,
              h_K_b_MW: float | None = None, rel_unc: float | None = 0.10):
    """A bonded joint: layers as (label, thickness_um, k_W_mK[, void_fraction[, material_key]]),
    from material_a to material_b; optional face conductances in MW/m^2K."""
    from .network.stack import FillSpec, Layer
    q = lambda v, u, n: Quantity.parse(v, u, rel_uncertainty=rel_unc,
                                       provenance=Provenance.USER_MEASURED, name=n)
    out = []
    for spec in layers:
        lab, t, k, *rest = spec
        f = rest[0] if rest else 0.0
        mk = rest[1] if len(rest) > 1 else None
        out.append(Layer(lab, q(t, "um", "t"), q(k, "W/mK", "k"), f, mk))
    hk = lambda v: None if v is None else q(v * 1e6, "W/m2K", "h_K")
    return FillSpec(kind="bond", layers=out, h_K_a=hk(h_K_a_MW), h_K_b=hk(h_K_b_MW))
