# LAMMPS reference for OpenKapitza (RAG corpus)

> **Generated — do not edit.** Rebuild with
> `python3 scripts/build_lammps_rag.py --lammps-src <clone>`.
>
> Source: official LAMMPS source tree, version **30 Sep 2026**.
> `docs.lammps.org` is built from these same `.rst` files, so this is
> the manual's content pinned to the engine actually in use. Example
> scripts are emitted verbatim; prose is quoted with its source named.

**Scope:** the physics paths this project uses — thermal transport, elastic constants and relaxation, hardness and indentation, gas–surface accommodation, surface structure, amorphous melt–quench, electron–phonon coupling, and gas transport properties. Bounded on purpose: a corpus that covers everything discriminates nothing.

**Retrieval note.** Every command carries its syntax, official examples, and all `note`/`warning` admonitions from its page. The admonitions matter most: they are where the manual records the constraints that otherwise surface as a failed run. LAMMPS line continuation is `&`, not a backslash — a lesson this project learned the hard way, and visible in `in.wall.ccl` below.

## Contents

1. [Thermal transport](#1-thermal-transport)
2. [Elastic constants and cell relaxation](#2-elastic-constants-and-cell-relaxation)
3. [Hardness and indentation](#3-hardness-and-indentation)
4. [Gas-surface scattering and accommodation coefficients](#4-gas-surface-scattering-and-accommodation-coefficients)
5. [Surface structure and roughness](#5-surface-structure-and-roughness)
6. [Amorphous structures and melt-quench](#6-amorphous-structures-and-melt-quench)
7. [Electron-phonon coupling (two-temperature model)](#7-electron-phonon-coupling-two-temperature-model)
8. [Gas transport properties](#8-gas-transport-properties)

---

## 1. Thermal transport

*Why this is here:* The Kapitza-resistance path. Five official methods for thermal conductivity; OpenKapitza uses method 1 (Langevin reservoirs) with non-periodic walls, and the temperature-fluctuation cross-check rather than Green-Kubo, which measures BULK conductivity.

### Official scripts

#### `KAPPA/in.langevin` — Method 1 - thermostat two slabs (what we use)

```lammps
# sample LAMMPS input script for thermal conductivity of liquid LJ
# thermostatting 2 regions via fix langevin

# settings
# (index-style variables can be overridden from the command line or a wrapper
# input for abbreviated runs, e.g. by the tests in the unittest tree)

variable        x index 10
variable        y index 10
variable        z equal 20

variable        rho equal 0.6
variable        t equal 1.35
variable        rc equal 2.5

variable        nequil index 1000
variable        nsteady index 10000
variable        nprod index 20000
variable        tlo equal 1.0
variable        thi equal 1.70

#variable       rho equal 0.85
#variable        t equal 0.7
#variable       rc equal 3.0
#variable        tlo equal 0.3
#variable        thi equal 1.0

# setup problem

units           lj
atom_style      atomic

lattice         fcc ${rho}
region          box block 0 $x 0 $y 0 $z
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create $t 87287

pair_style      lj/cut ${rc}
pair_coeff      1 1 1.0 1.0

neighbor        0.3 bin
neigh_modify    delay 0 every 1

# heat layers

region          hot block INF INF INF INF 0 1
region          cold block  INF INF INF INF 10 11
compute         Thot all temp/region hot
compute         Tcold all temp/region cold

# 1st equilibration run

fix             1 all nvt temp $t $t 0.5
thermo          100
run             ${nequil}

velocity        all scale $t

unfix           1

# 2nd equilibration run

fix             1 all nve
fix             hot all langevin ${thi} ${thi} 1.0 59804 tally yes
fix             cold all langevin ${tlo} ${tlo} 1.0 287859 tally yes
fix_modify      hot temp Thot
fix_modify      cold temp Tcold

variable        tdiff equal c_Thot-c_Tcold
thermo_style    custom step temp c_Thot c_Tcold f_hot f_cold v_tdiff
thermo_modify   colname c_Thot Temp_hot colname c_Tcold Temp_cold &
                colname f_hot E_hot colname f_cold E_cold &
                colname v_tdiff dTemp_step
thermo          1000
run             ${nsteady}

# thermal conductivity calculation
# reset langevin thermostats to zero energy accumulation

compute         ke all ke/atom
variable        temp atom c_ke/1.5

fix             hot all langevin ${thi} ${thi} 1.0 59804 tally yes
fix             cold all langevin ${tlo} ${tlo} 1.0 287859 tally yes
fix_modify      hot temp Thot
fix_modify      cold temp Tcold

fix             ave all ave/time 10 100 1000 v_tdiff ave running
thermo_style    custom step temp c_Thot c_Tcold f_hot f_cold v_tdiff f_ave
thermo_modify   colname c_Thot Temp_hot colname c_Tcold Temp_cold &
                colname f_hot E_hot colname f_cold E_cold &
                colname v_tdiff dTemp_step colname f_ave dTemp

compute         layers all chunk/atom bin/1d z lower 0.05 units reduced
fix             2 all ave/chunk 10 100 1000 layers v_temp file profile.langevin

variable start_time equal time
variable kappa equal (0.5*(abs(f_hot)+abs(f_cold))/(time-${start_time})/(lx*ly)/2.0)*(lz/2.0)/f_ave

run             ${nprod}
print           "Running average thermal conductivity: $(v_kappa:%.2f)"
```

#### `KAPPA/in.heat` — Method 2 - add/subtract energy with fix heat

```lammps
# sample LAMMPS input script for thermal conductivity of liquid LJ
# use fix heat to add/subtract energy from 2 regions

# settings
# (index-style variables can be overridden from the command line or a wrapper
# input for abbreviated runs, e.g. by the tests in the unittest tree)

variable        x index 10
variable        y index 10
variable        z equal 20

variable        rho equal 0.6
variable        t equal 1.35
variable        rc equal 2.5

variable        nequil index 1000
variable        nsteady index 10000
variable        nprod index 20000

#variable       rho equal 0.85
#variable        t equal 0.7
#variable       rc equal 3.0

# setup problem

units           lj
atom_style      atomic

lattice         fcc ${rho}
region          box block 0 $x 0 $y 0 $z
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create $t 87287

pair_style      lj/cut ${rc}
pair_coeff      1 1 1.0 1.0

neighbor        0.3 bin
neigh_modify    delay 0 every 1

# heat layers

region          hot block INF INF INF INF 0 1
region          cold block  INF INF INF INF 10 11
compute         Thot all temp/region hot
compute         Tcold all temp/region cold

# 1st equilibration run

fix             1 all nvt temp $t $t 0.5
thermo          100
run             ${nequil}

velocity        all scale $t

unfix           1

# 2nd equilibration run

fix             1 all nve
fix             hot all heat 1 100.0 region hot
fix             cold all heat 1 -100.0 region cold

thermo_style    custom step temp c_Thot c_Tcold
thermo_modify   colname c_Thot Temp_hot colname c_Tcold Temp_cold
thermo          1000
run             ${nsteady}

# thermal conductivity calculation

compute         ke all ke/atom
variable        temp atom c_ke/1.5

compute         layers all chunk/atom bin/1d z lower 0.05 units reduced
fix             2 all ave/chunk 10 100 1000 layers v_temp file profile.heat

variable        tdiff equal f_2[1][3]-f_2[11][3]
fix             ave all ave/time 1 1 1000 v_tdiff ave running start $(v_nequil+v_nsteady+v_nprod/10)

variable kappa equal (100/(lx*ly)/2.0)*(lz/2.0)/f_ave
thermo_style    custom step temp c_Thot c_Tcold v_tdiff f_ave
thermo_modify   colname c_Thot Temp_hot colname c_Tcold Temp_cold &
                colname v_tdiff  dTemp_step colname f_ave dTemp

run             ${nprod}
print           "Running average thermal conductivity: $(v_kappa:%.2f)"
```

#### `KAPPA/in.ehex` — Method 3 - enhanced heat exchange

```lammps
# sample LAMMPS input script for thermal conductivity of liquid LJ
# use fix ehex to add/subtract energy from 2 regions

# settings
# (index-style variables can be overridden from the command line or a wrapper
# input for abbreviated runs, e.g. by the tests in the unittest tree)

variable        x index 10
variable        y index 10
variable        z equal 20

variable        rho equal 0.6
variable        t equal 1.35
variable        rc equal 2.5

variable        nequil index 1000
variable        nsteady index 10000
variable        nprod index 20000

#variable       rho equal 0.85
#variable        t equal 0.7
#variable       rc equal 3.0

# setup problem

units           lj
atom_style      atomic

lattice         fcc ${rho}
region          box block 0 $x 0 $y 0 $z
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create $t 87287

pair_style      lj/cut ${rc}
pair_coeff      1 1 1.0 1.0

neighbor        0.3 bin
neigh_modify    delay 0 every 1

# heat layers

region          hot block INF INF INF INF 0 1
region          cold block  INF INF INF INF 10 11
compute         Thot all temp/region hot
compute         Tcold all temp/region cold

# 1st equilibration run

fix             1 all nvt temp $t $t 0.5
thermo          100
run             ${nequil}

velocity        all scale $t

unfix           1

# 2nd equilibration run

fix             1 all nve
fix             hot all ehex 1 100.0 region hot
fix             cold all ehex 1 -100.0 region cold

thermo_style    custom step temp c_Thot c_Tcold
thermo_modify   colname c_Thot Temp_hot colname c_Tcold Temp_cold
thermo          1000
run             ${nsteady}

# thermal conductivity calculation

compute         ke all ke/atom
variable        temp atom c_ke/1.5

compute         layers all chunk/atom bin/1d z lower 0.05 units reduced
fix             2 all ave/chunk 10 100 1000 layers v_temp file profile.ehex

variable        tdiff equal f_2[1][3]-f_2[11][3]
fix             ave all ave/time 1 1 1000 v_tdiff ave running start $(v_nequil+v_nsteady+v_nprod/10)

variable kappa equal (100/(lx*ly)/2.0)*(lz/2.0)/f_ave

thermo_style    custom step temp c_Thot c_Tcold v_tdiff f_ave
thermo_modify   colname c_Thot Temp_hot colname c_Tcold Temp_cold &
                colname v_tdiff dTemp_step colname f_ave dTemp

run             ${nprod}
print           "Running average thermal conductivity: $(v_kappa:%.2f)"
```

#### `KAPPA/in.mp` — Method 4 - Muller-Plathe reverse NEMD

```lammps
# sample LAMMPS input script for thermal conductivity of liquid LJ
# Muller-Plathe method via fix thermal_conductivity

# settings
# (index-style variables can be overridden from the command line or a wrapper
# input for abbreviated runs, e.g. by the tests in the unittest tree)

variable        x index 10
variable        y index 10
variable        z equal 20

variable        rho equal 0.6
variable        t equal 1.35
variable        rc equal 2.5

variable        nequil index 1000
variable        nsteady index 20000
variable        nprod index 20000

#variable       rho equal 0.85
#variable        t equal 0.7
#variable       rc equal 3.0

# setup problem

units           lj
atom_style      atomic

lattice         fcc ${rho}
region          box block 0 $x 0 $y 0 $z
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create $t 87287

pair_style      lj/cut ${rc}
pair_coeff      1 1 1.0 1.0

neighbor        0.3 bin
neigh_modify    delay 0 every 1

# 1st equilibration run

fix             1 all nvt temp $t $t 0.5
thermo          100
run             ${nequil}

velocity        all scale $t

unfix           1

# 2nd equilibration run

compute         ke all ke/atom
variable        temp atom c_ke/1.5

fix             1 all nve

compute         layers all chunk/atom bin/1d z lower 0.05 units reduced
fix             2 all ave/chunk 10 100 1000 layers v_temp file profile.mp
fix             3 all thermal/conductivity 10 z 20

variable        tdiff equal f_2[11][3]-f_2[1][3]
thermo_style    custom step temp epair etotal f_3 v_tdiff
thermo_modify   colname f_3 E_delta colname v_tdiff dTemp_step

thermo          1000
run             ${nsteady}

# thermal conductivity calculation
# reset fix thermal/conductivity to zero energy accumulation
fix             3 all thermal/conductivity 10 z 20

variable start_time equal time
variable kappa equal (f_3/(time-${start_time})/(lx*ly)/2.0)*(lz/2.0)/f_ave

fix             ave all ave/time 1 1 1000 v_tdiff ave running
thermo_style    custom step temp epair etotal f_3 v_tdiff f_ave
thermo_modify   colname f_3 E_delta colname v_tdiff dTemp_step colname f_ave dTemp

run             ${nprod}
print           "Running average thermal conductivity: $(v_kappa:%.2f)"
```

#### `KAPPA/in.heatflux` — Method 5 - Green-Kubo equilibrium

```lammps
# sample LAMMPS input script for thermal conductivity of liquid LJ
# Green-Kubo method via compute heat/flux and fix ave/correlate

# settings
# (index-style variables can be overridden from the command line or a wrapper
# input for abbreviated runs, e.g. by the tests in the unittest tree)

variable        x index 10
variable        y index 10
variable        z equal 10

variable        rho equal 0.6
variable        t equal 1.35
variable        rc equal 2.5

variable        nequil index 1000
variable        nprod index 100000

#variable       rho equal 0.85
#variable        t equal 0.7
#variable       rc equal 3.0

variable    p index 200     # correlation length
variable    s index 10      # sample interval
variable    d equal $p*$s   # dump interval

# setup problem

units           lj
atom_style      atomic

lattice         fcc ${rho}
region          box block 0 $x 0 $y 0 $z
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create $t 87287

pair_style      lj/cut ${rc}
pair_coeff      1 1 1.0 1.0

neighbor        0.3 bin
neigh_modify    delay 0 every 1

# 1st equilibration run

fix             1 all nvt temp $t $t 0.5
thermo          100
run             ${nequil}

velocity        all scale $t

unfix           1

# thermal conductivity calculation

reset_timestep  0

compute         myKE all ke/atom
compute         myPE all pe/atom
compute         myStress all stress/atom NULL virial
compute         flux all heat/flux myKE myPE myStress
variable        Jx equal c_flux[1]/vol
variable        Jy equal c_flux[2]/vol
variable        Jz equal c_flux[3]/vol

fix             1 all nve
fix             JJ all ave/correlate $s $p $d &
                c_flux[1] c_flux[2] c_flux[3] type auto &
                file profile.heatflux ave running

variable        scale equal $s*dt/$t/$t/vol
variable        k11 equal trap(f_JJ[3])*${scale}
variable        k22 equal trap(f_JJ[4])*${scale}
variable        k33 equal trap(f_JJ[5])*${scale}
variable        kappa equal (v_k11+v_k22+v_k33)/3.0

thermo          $d
thermo_style    custom step temp v_Jx v_Jy v_Jz v_k11 v_k22 v_k33 v_kappa
thermo_modify   colname v_Jx Jx colname v_Jy Jy colname v_Jz Jz &
                colname v_k11 kappa_11 colname v_k22 kappa_22 &
                colname v_k33 kappa_33 colname v_kappa kappa

run             ${nprod}

print           "Running average thermal conductivity: $(v_kappa:%.2f)"
```

### Command reference

#### `fix langevin` — Langevin reservoirs

*Source: `doc/src/fix_langevin.rst`*

```lammps
fix ID group-ID langevin Tstart Tstop damp seed keyword values ...
```

```lammps
*angmom* value = *no* or factor
  *no* = do not thermostat rotational degrees of freedom via the angular momentum
  factor = do thermostat rotational degrees of freedom via the angular momentum and apply numeric scale factor as discussed below
*omega* value = *no* or *yes*
  *no* = do not thermostat rotational degrees of freedom via the angular velocity
  *yes* = do thermostat rotational degrees of freedom via the angular velocity
*scale* values = type ratio
  type = atom type (1-N)
  ratio = factor by which to scale the damping coefficient
*tally* value = *no* or *yes*
  *no* = do not tally the energy added/subtracted to atoms
  *yes* = do tally the energy added/subtracted to atoms
*zero* value = *no* or *yes*
  *no* = do not set total random force to zero
  *yes* = set total random force to zero
```

* ID, group-ID are documented in `fix` command
* langevin = style name of this fix command
* Tstart,Tstop = desired temperature at start/end of run (temperature units)
* Tstart can be a variable (see below)
* damp = damping parameter (time units)
* seed = random number seed to use for white noise (positive integer)
* zero or more keyword/value pairs may be appended
* keyword = *angmom* or *omega* or *scale* or *tally* or *zero*
       *angmom* value = *no* or factor
         *no* = do not thermostat rotational degrees of freedom via the angular momentum
       *omega* value = *no* or *yes*
         *no* = do not thermostat rotational degrees of freedom via the angular velocity

*Examples*

```lammps
fix 3 boundary langevin 1.0 1.0 1000.0 699483
fix 1 all langevin 1.0 1.1 100.0 48279 scale 3 1.5
fix 1 all langevin 1.0 1.1 100.0 48279 angmom 3.333
```

Apply a Langevin thermostat as described in (Bruenger)
to a group of atoms which models an interaction with a background
implicit solvent.  Used with `fix nve`, this command
performs Brownian dynamics (BD), since the total force on each atom
will have the form:

F = & F_c + F_f + F_r \\
   F_f = & - \frac{m}{\mathrm{damp}} v \\
   F_r \propto & \sqrt{\frac{k_B T m}{dt~\mathrm{damp}}}

*Constraints and cautions from this page*

- **NOTE:** Unlike the `fix nvt` command which performs Nose/Hoover thermostatting AND time integration, this fix does NOT perform time integration.  It only modifies forces to effect thermostatting.  Thus you must use a separate time integration fix, like `fix nve` to actually update the velocities and positions of atoms using the modified forces.  Likewise, this fix should not normally be used on atoms that also have their temperature controlled by another fix - e.g. by `fix nvt` or `fix temp/rescale` commands.

#### `fix heat` — add/subtract energy

*Source: `doc/src/fix_heat.rst`*

```lammps
fix ID group-ID heat N eflux
```

```lammps
*region* value = region-ID
  region-ID = ID of region atoms must be in to have added force
```

* ID, group-ID are documented in `fix` command
* heat = style name of this fix command
* N = add/subtract heat every this many timesteps
* eflux = rate of heat addition or subtraction (energy/time units)
* eflux can be a variable (see below)
* zero or more keyword/value pairs may be appended to args
* keyword = *region*
       *region* value = region-ID

*Examples*

```lammps
fix 3 qin heat 1 1.0
fix 3 qin heat 10 v_flux
fix 4 qout heat 1 -1.0 region top
```

Add non-translational kinetic energy (heat) to a group of atoms in a
manner that conserves their aggregate momentum.  Two of these fixes
can be used to establish a temperature gradient across a simulation
domain by adding heat (energy) to one group of atoms (hot reservoir)
and subtracting heat from another (cold reservoir).  E.g. a simulation
sampling from the McDLT ensemble.

If the *region* keyword is used, the atom must be in both the group
and the specified geometric `region` in order to have
energy added or subtracted to it.  If not specified, then the atoms in
the group are affected wherever they may move to.

*Constraints and cautions from this page*

- **NOTE:** The heat-exchange (HEX) algorithm implemented by this fix is known to exhibit a pronounced energy drift. An improved algorithm (eHEX) is available as a `fix ehex` command and might be preferable if energy conservation is important.
- **NOTE:** If heat is subtracted from the system too aggressively so that the group's kinetic energy would go to zero, or any individual atom's kinetic energy would go to zero for the case where *eflux* is an atom-style variable, then LAMMPS will halt with an error message.

#### `fix ehex` — enhanced heat exchange

*Source: `doc/src/fix_ehex.rst`*

```lammps
fix ID group-ID ehex nevery F keyword value
```

```lammps
*region* value = region-ID
  region-ID = ID of region (reservoir) atoms must be in for added thermostatting force
*constrain* value = none
  apply the constraint algorithm (SHAKE or RATTLE) again at the end of the timestep
*com* value = none
  rescale all sites of a constrained cluster of atom if its COM is in the reservoir
*hex* value = none
  omit the coordinate correction to recover the HEX algorithm
```

* ID, group-ID are documented in `fix` command
* ehex  = style name of this fix command
* nevery = add/subtract heat every this many timesteps
* F = energy flux into the reservoir (energy/time units)
* zero or more keyword/value pairs may be appended to args
* keyword = *region* or *constrain* or *com* or *hex*
       *region* value = region-ID
       *constrain* value = none
       *com* value = none
       *hex* value = none

*Examples*

```lammps
# Lennard-Jones, examples/HEAT/in.lj.ehex

fix fnve all nve
# specify regions rhot and rcold
...
fix fhot all ehex 1 0.15 region rhot
fix fcold all ehex 1 -0.15 region rcold

# SPC/E water, from examples/HEAT/in.spce.ehex
fix fnve all nve
# specify regions rhot and rcold
...
fix fhot all ehex 1 0.075 region rhot constrain com
fix fcold all ehex 1 -0.075 region rcold constrain com
fix frattle all rattle 1e-10 400 0 b 1 a 1
```

This fix implements the asymmetric version of the enhanced heat
exchange algorithm (Wirnsberger). The eHEX algorithm is
an extension of the heat exchange algorithm (Ikeshoji2) and
adds an additional coordinate integration to account for higher-order
truncation terms in the operator splitting.  The original HEX
algorithm (implemented as `fix heat`) is known to
exhibit a slight energy drift limiting the accessible simulation times
to a few nanoseconds.  This issue is greatly improved by the new
algorithm decreasing the energy drift by at least a factor of a
hundred (LJ and SPC/E water) with little computational overhead.

In both algorithms (non-translational) kinetic energy is constantly
swapped between regions (reservoirs) to impose a heat flux onto the
system.  The equations of motion are therefore modified if a particle
$i$ is located inside a reservoir $\Gamma_k$ where $k>0$.  We
use $\Gamma_0$ to label those parts of the simulation box which
are not thermostatted.)  The input parameter *region-ID* of this fix
corresponds to $k$.  The energy swap is modelled by introducing an
additional thermostatting force to the equations of motion, such that
the time evolution of coordinates and momenta of particle $i$
becomes (Wirnsberger)

*Constraints and cautions from this page*

- **NOTE:** This fix only integrates the thermostatting force and must be combined with another integrator, such as `fix nve`, to solve the full equations of motion.
- **NOTE:** If heat is subtracted from the system too aggressively so that the group's kinetic energy would go to zero, then LAMMPS will halt with an error message.  Increasing the value of *nevery* means that heat is added/subtracted less frequently but in larger portions.  The resulting temperature profile will therefore be the same.
- **NOTE:** You can only use the keyword *com* along with *constrain*.
- **NOTE:** Even if RATTLE is used and the keywords *com* and *constrain* are both set, the coordinate constraints will not necessarily be satisfied up to the target precision. The velocity constraints are satisfied as long as all sites of a cluster are rescaled (keyword *com*) and the cluster does not span adjacent reservoirs. The current implementation of the eHEX algorithm introduces a small error in the bond distances, which goes to zero with order three in the timestep. For example, in a simulation of SPC/E water with a timestep of 2 fs the maximum relative error in the bond distances was found to be on the order of $10^{-7}$ for relatively large temperature gradients.  A higher precision can be achieved by decreasing the timestep.

#### `fix thermal/conductivity` — Muller-Plathe reverse NEMD

*Source: `doc/src/fix_thermal_conductivity.rst`*

```lammps
fix ID group-ID thermal/conductivity N edim Nbin keyword value ...
```

```lammps
*swap* value = Nswap = number of swaps to perform every N steps
```

* ID, group-ID are documented in `fix` command
* thermal/conductivity = style name of this fix command
* N = perform kinetic energy exchange every N steps
* edim = *x* or *y* or *z* = direction of kinetic energy transfer
* Nbin = # of layers in edim direction (must be even number)
* zero or more keyword/value pairs may be appended
* keyword = *swap*
       *swap* value = Nswap = number of swaps to perform every N steps

*Examples*

```lammps
fix 1 all thermal/conductivity 100 z 20
fix 1 all thermal/conductivity 50 z 20 swap 2
```

Use the Muller-Plathe algorithm described in this paper to exchange kinetic energy between two particles
in different regions of the simulation box every N steps.  This
induces a temperature gradient in the system.  As described below this
enables the thermal conductivity of a material to be calculated.  This
algorithm is sometimes called a reverse non-equilibrium MD (reverse
NEMD) approach to computing thermal conductivity.  This is because the
usual NEMD approach is to impose a temperature gradient on the system
and measure the response as the resulting heat flux.  In the
Muller-Plathe method, the heat flux is imposed, and the temperature
gradient is the system's response.

See the `compute heat/flux` command for details
on how to compute thermal conductivity in an alternate way, via the
Green-Kubo formalism.

*Constraints and cautions from this page*

- **NOTE:** If your system is periodic in the direction of the heat flux, then the flux is going in 2 directions.  This means the effective heat flux in one direction is reduced by a factor of 2.  You will see this in the equations for thermal conductivity (kappa) in the Muller-Plathe paper.  LAMMPS is simply tallying kinetic energy which does not account for whether or not your system is periodic; you must use the value appropriately to yield a kappa for your system.
- **NOTE:** After equilibration, if the temperature gradient you observe is not linear, then you are likely swapping energy too frequently and are not in a regime of linear response.  In this case you cannot accurately infer a thermal conductivity and should try increasing the Nevery parameter.

#### `compute heat/flux` — Green-Kubo heat flux

*Source: `doc/src/compute_heat_flux.rst`*

```lammps
compute ID group-ID heat/flux ke-ID pe-ID stress-ID
```

* ID, group-ID are documented in `compute` command
* heat/flux = style name of this compute command
* ke-ID = ID of a compute that calculates per-atom kinetic energy
* pe-ID = ID of a compute that calculates per-atom potential energy
* stress-ID = ID of a compute that calculates per-atom stress

*Examples*

```lammps
compute myFlux all heat/flux myKE myPE myStress
```

Define a computation that calculates the heat flux vector based on
contributions from atoms in the specified group.  This can be used by
itself to measure the heat flux through a set of atoms (e.g., a region
between two thermostatted reservoirs held at different temperatures),
or to calculate a thermal conductivity using the equilibrium
Green-Kubo formalism.

For other non-equilibrium ways to compute a thermal conductivity, see
the `Howto kappa` doc page.  These include use of
the `fix thermal/conductivity` command
for the Muller-Plathe method.  Or the `fix heat` command
which can add or subtract heat from groups of atoms.

*Constraints and cautions from this page*

- **NOTE:** These other computes should provide values for all the atoms in the group this compute specifies.  That means the other computes could use the same group as this compute, or they can just use group "all" (or any group whose atoms are superset of the atoms in this compute's group).  LAMMPS does not check for this.
- **NOTE:** The `compute pe/atom` and `compute stress/atom` commands have options for which terms to include in their calculation (pair, bond, etc).  The heat flux calculation will thus include exactly the same terms. Normally you should use `compute stress/atom virial` or `compute centroid/stress/atom virial` so as not to include a kinetic energy term in the heat flux.
- **WARNING:** The compute *heat/flux* has been reported to produce unphysical values for angle, dihedral, improper and constraint force contributions when used with `compute stress/atom`, as discussed in (Surblys2019), (Boone) and (Surblys2021). You are strongly advised to use `compute centroid/stress/atom`, which has been implemented specifically for such cases.
- **WARNING:** Due to an implementation detail, the $y$ and $z$ components of heat flux from `fix rigid` contribution when computed via `compute stress/atom` are highly unphysical and should not be used.

#### `compute temp/region` — temperature of a spatial slab

*Source: `doc/src/compute_temp_region.rst`*

```lammps
compute ID group-ID temp/region region-ID
```

* ID, group-ID are documented in `compute` command
* temp/region = style name of this compute command
* region-ID = ID of region to use for choosing atoms

*Examples*

```lammps
compute mine flow temp/region boundary
```

Define a computation that calculates the temperature of a group of atoms in a
geometric region.  This can be useful for thermostatting one portion of the
simulation box.  For example, a McDLT simulation where one side is cooled, and
the other side is heated.  A compute of this style can be used by any command
that computes a temperature (e.g., `thermo_modify`,
`fix temp/rescale`).

Note that a *region*-style temperature can be used to thermostat with
`fix temp/rescale` or
`fix langevin`, but should probably not be used with
Nose--Hoover style fixes (`fix nvt`, `fix npt`,
or `fix nph`) if the degrees of freedom included in the computed
temperature vary with time.

#### `compute ke/atom` — per-atom KE for profiles

*Source: `doc/src/compute_ke_atom.rst`*

```lammps
compute ID group-ID ke/atom
```

* ID, group-ID are documented in `compute` command
* ke/atom = style name of this compute command

*Examples*

```lammps
compute 1 all ke/atom
```

Define a computation that calculates the per-atom translational
kinetic energy for each atom in a group.

The kinetic energy is simply $\frac12 m v^2$, where $m$ is the mass
and $v$ is the velocity of each atom.

#### `compute chunk/atom` — binning along an axis

*Source: `doc/src/compute_chunk_atom.rst`*

```lammps
compute ID group-ID chunk/atom style args keyword values ...
```

```lammps
style = *bin/1d* or *bin/2d* or *bin/3d* or *bin/sphere* or *bin/cylinder* or *type* or *molecule* or c_ID, c_ID[I], f_ID, f_ID[I], v_name
  *bin/1d* args = dim origin delta
    dim = *x* or *y* or *z*
    origin = *lower* or *center* or *upper* or coordinate value (distance units)
    delta = thickness of spatial bins in dim (distance units)
  *bin/2d* args = dim origin delta dim origin delta
    dim = *x* or *y* or *z*
    origin = *lower* or *center* or *upper* or coordinate value (distance units)
    delta = thickness of spatial bins in dim (distance units)
  *bin/3d* args = dim origin delta dim origin delta dim origin delta
    dim = *x* or *y* or *z*
    origin = *lower* or *center* or *upper* or coordinate value (distance units)
    delta = thickness of spatial bins in dim (distance units)
  *bin/sphere* args = xorig yorig zorig rmin rmax nsbin
    xorig,yorig,zorig = center point of sphere
    srmin,srmax = bin from sphere radius rmin to rmax
    nsbin = # of spherical shell bins between rmin and rmax
  *bin/cylinder* args = dim origin delta c1 c2 rmin rmax ncbin
    dim = *x* or *y* or *z* = axis of cylinder axis
    origin = *lower* or *center* or *upper* or coordinate value (distance units)
    delta = thickness of spatial bins in dim (distance units)
    c1,c2 = coords of cylinder axis in other 2 dimensions (distance units)
    crmin,crmax = bin from cylinder radius rmin to rmax (distance units)
    ncbin = # of concentric circle bins between rmin and rmax
  *type* args = none
  *molecule* args = none
  c_ID, c_ID[I], f_ID, f_ID[I], v_name args = none
    c_ID = per-atom vector calculated by a compute with ID
    c_ID[I] = Ith column of per-atom array calculated by a compute with ID
    f_ID = per-atom vector calculated by a fix with ID
    f_ID[I] = Ith column of per-atom array calculated by a fix with ID
    v_name = per-atom vector calculated by an atom-style variable with name
```

```lammps
*region* value = region-ID
  region-ID = ID of region atoms must be in to be part of a chunk
*nchunk* value = *once* or *every*
  once = only compute the number of chunks once
  every = re-compute the number of chunks whenever invoked
*limit* values = 0 or Nc max or Nc exact
  0 = no limit on the number of chunks
  Nc max = limit number of chunks to be <= Nc
  Nc exact = set number of chunks to exactly Nc
*ids* value = *once* or *nfreq* or *every*
  once = assign chunk IDs to atoms only once, they persist thereafter
  nfreq = assign chunk IDs to atoms only once every Nfreq steps (if invoked by :doc:`fix ave/chunk <fix_ave_chunk>` which sets Nfreq)
  every = assign chunk IDs to atoms whenever invoked
*compress* value = *yes* or *no*
  yes = compress chunk IDs to eliminate IDs with no atoms
  no = do not compress chunk IDs even if some IDs have no atoms
*discard* value = *yes* or *no* or *mixed*
  yes = discard atoms with out-of-range chunk IDs by assigning a chunk ID = 0
  no = keep atoms with out-of-range chunk IDs by assigning a valid chunk ID
  mixed = keep or discard such atoms according to spatial binning rule
*bound* values = x/y/z lo hi
  x/y/z = *x* or *y* or *z* to bound spatial bins in this dimension
  lo = *lower* or coordinate value (distance units)
  hi = *upper* or coordinate value (distance units)
*pbc* value = *no* or *yes*
  yes = use periodic distance for bin/sphere and bin/cylinder styles
*units* value = *box* or *lattice* or *reduced*
```

* ID, group-ID are documented in `compute` command
* chunk/atom = style name of this compute command
       *bin/1d* args = dim origin delta
       *bin/2d* args = dim origin delta dim origin delta
       *bin/3d* args = dim origin delta dim origin delta dim origin delta
       *bin/sphere* args = xorig yorig zorig rmin rmax nsbin
       *bin/cylinder* args = dim origin delta c1 c2 rmin rmax ncbin
       *type* args = none
       *molecule* args = none
* zero or more keyword/values pairs may be appended
* keyword = *region* or *nchunk* or *limit* or *ids* or *compress* or *discard* or *bound* or *pbc* or *units*
       *region* value = region-ID

*Examples*

```lammps
compute 1 all chunk/atom type
compute 1 all chunk/atom bin/1d z lower 0.02 units reduced
compute 1 all chunk/atom bin/2d z lower 1.0 y 0.0 2.5
compute 1 all chunk/atom molecule region sphere nchunk once ids once compress yes
compute 1 all chunk/atom bin/sphere 5 5 5 2.0 5.0 5 discard yes
compute 1 all chunk/atom bin/cylinder z lower 2 10 10 2.0 5.0 3 discard yes
compute 1 all chunk/atom c_cluster
```

Define a computation that calculates an integer chunk ID from 1 to
Nchunk for each atom in the group.  Values of chunk IDs are determined
by the *style* of chunk, which can be based on atom type or molecule
ID or spatial binning or a per-atom property or value calculated by
another `compute`, `fix`, or `atom-style variable`.  Per-atom chunk IDs can be used by other
computes with "chunk" in their style name, such as `compute com/chunk` or `compute msd/chunk`.  Or they can be used by the `fix ave/chunk` command to sum and time average a
variety of per-atom properties over the atoms in each chunk.  Or they
can simply be accessed by any command that uses per-atom values from a
compute as input, as discussed on the `Howto output`
doc page.

See the `Howto chunk` page for an overview of how
this compute can be used with a variety of other commands to tabulate
properties of a simulation.  The page gives several examples of input
script commands that can be used to calculate interesting properties.

*Constraints and cautions from this page*

- **NOTE:** If you want the persistent chunk-IDs calculated by this compute to be continuous when running from a `restart file`, then you should use the same ID for this compute, as in the original run.  This is so that the fix this compute creates to store per-atom quantities will also have the same ID, and thus be initialized correctly with chunk IDs from the restart file.
- **NOTE:** The compression operation requires global communication across all processors to share their chunk ID values.  It can require large memory on every processor to store them, even after they are compressed, if there are a large number of unique chunk IDs with atoms assigned to them.  It uses a STL map to find unique chunk IDs and store them in sorted order.  Each time an atom is assigned a compressed chunk ID, it must access the STL map.  All of this means that compression can be expensive, both in memory and CPU time.  The use of the *limit* keyword in conjunction with the *compress* keyword can affect these costs, depending on which keyword is used first.  So use this option with care.

#### `fix ave/chunk` — temperature profile

*Source: `doc/src/fix_ave_chunk.rst`*

```lammps
fix ID group-ID ave/chunk Nevery Nrepeat Nfreq chunkID value1 value2 ... keyword args ...
```

```lammps
vx,vy,vz,fx,fy,fz,mass = atom attribute (velocity, force component, mass)
density/number, density/mass = number or mass density (per volume)
temp = temperature
c_ID = per-atom vector calculated by a compute with ID
c_ID[I] = Ith column of per-atom array calculated by a compute with ID, I can include wildcard (see below)
f_ID = per-atom vector calculated by a fix with ID
f_ID[I] = Ith column of per-atom array calculated by a fix with ID, I can include wildcard (see below)
v_name = per-atom vector calculated by an atom-style variable with name
```

```lammps
*norm* arg = *all* or *sample* or *none* = how output on *Nfreq* steps is normalized
  all = output is sum of atoms across all *Nrepeat* samples, divided by atom count
  sample = output is sum of *Nrepeat* sample averages, divided by *Nrepeat*
  none = output is sum of *Nrepeat* sample sums, divided by *Nrepeat*
*ave* args = *one* or *running* or *window M*
  one = output new average value every Nfreq steps
  running = output cumulative average of all previous Nfreq steps
  window M = output average of M most recent Nfreq steps
*bias* arg = bias-ID
  bias-ID = ID of a temperature compute that removes a velocity bias for temperature calculation
*adof* value = dof_per_atom
  dof_per_atom = define this many degrees-of-freedom per atom for temperature calculation
*cdof* value = dof_per_chunk
  dof_per_chunk = define this many degrees-of-freedom per chunk for temperature calculation
*file* arg = filename
  filename = file to write results to
*append* arg = filename
  filename = file to append results to
*overwrite* arg = none = overwrite output file with only latest output
*format* arg = string
  string = C-style format string
*title1* arg = string
  string = text to print as 1st line of output file
*title2* arg = string
  string = text to print as 2nd line of output file
*title3* arg = string
  string = text to print as 3rd line of output file
```

* ID, group-ID are documented in `fix` command
* ave/chunk = style name of this fix command
* Nevery = use input values every this many timesteps
* Nrepeat = # of times to use input values for calculating averages
* Nfreq = calculate averages every this many timesteps
* chunkID = ID of `compute chunk/atom` command
* one or more input values can be listed
* value = *vx*, *vy*, *vz*, *fx*, *fy*, *fz*, *density/mass*, *density/number*, *mass*, *temp*, c_ID, c_ID[I], f_ID, f_ID[I], v_name
* zero or more keyword/arg pairs may be appended
* keyword = *norm* or *ave* or *bias* or *adof* or *cdof* or *file* or *append* or *overwrite* or *format* or *title1* or *title2* or *title3*
       *norm* arg = *all* or *sample* or *none* = how output on *Nfreq* steps is normalized
       *ave* args = *one* or *running* or *window M*

*Examples*

```lammps
fix 1 all ave/chunk 10000 1 10000 binchunk c_myCentro title1 "My output values"
fix 1 flow ave/chunk 100 10 1000 molchunk vx vz norm sample file vel.profile
fix 1 flow ave/chunk 100 5 1000 binchunk density/mass ave running
fix 1 flow ave/chunk 100 5 1000 binchunk density/mass ave running
```

Use one or more per-atom vectors as inputs every few timesteps, sum
the values over the atoms in each chunk at each timestep, then average
the per-chunk values over longer timescales.  The resulting chunk
averages can be used by other `output commands` such
as `thermo_style custom`, and can also be written to
a file.

In LAMMPS, chunks are collections of atoms defined by a `compute
chunk/atom` command, which assigns each atom to a
single chunk (or no chunk).  The ID for this command is specified as
chunkID.  For example, a single chunk could be the atoms in a molecule
or atoms in a spatial bin.  See the `compute chunk/atom` page and the `Howto chunk`
page for details of how chunks can be defined and examples of how they
can be used to measure properties of a system.

*Constraints and cautions from this page*

- **NOTE:** .. versionchanged:: 31May2016 If you are trying to replace a deprecated fix ave/spatial command with the newer, more flexible fix ave/chunk and `compute chunk/atom` commands, you simply need to split the fix ave/spatial arguments across the two new commands.  For example, this command: .. code-block:: LAMMPS fix 1 flow ave/spatial 100 10 1000 y 0.0 1.0 vx vz norm sample file vel.profile could be replaced by: .. code-block:: LAMMPS compute cc1 flow chunk/atom bin/1d y 0.0 1.0 fix 1 flow ave/chunk 100 10 1000 cc1 vx vz norm sample file vel.profile
- **NOTE:** This fix works by creating an array of size $N_\text{chunk} \times N_\text{values}$ on each processor. $N_\text{chunk}$ is the number of chunks, which is defined by the `compute chunk/atom` command. $N_\text{values}$ is the number of input values specified. Each processor loops over its atoms, tallying its values to the appropriate chunk.  Then the entire array is summed across all processors.  This means that using a large number of chunks will incur an overhead in memory and computational cost (summing across processors), so be careful to define a reasonable number of chunks.
- **NOTE:** To perform per-chunk averaging within a $N_\text{freq}$ time window, the number of chunks $N_\text{chunk}$ defined by the `compute chunk/atom` command must remain constant.  If the *ave* keyword is set to *running* or *window* then $N_\text{chunk}$ must remain constant for the duration of the simulation.  This fix forces the chunk/atom compute specified by chunkID to hold $N_\text{chunk}$ constant for the appropriate time windows, by not allowing it to re-calculate $N_\text{chunk}$, which can also affect how it assigns chunk IDs to atoms.  This is particularly important to understand if the chunks defined by the `compute chunk/atom` command are spatial bins.  If its *units* keyword is set to *box* or *lattice*, then the number of bins $N_\text{chunk}$ and size of each bin will be fixed over the $N_\text{freq}$ time window, which can affect which atoms are discarded if the simulation box size changes.  If its *units* keyword is set to *reduced*, then the number of bins $N_\text{chunk}$ will still be fixed, but the size of each bin can vary at each time step if the simulation box size changes (e.g., for an NPT simulation).

#### `fix ave/time` — running averages

*Source: `doc/src/fix_ave_time.rst`*

```lammps
fix ID group-ID ave/time Nevery Nrepeat Nfreq value1 value2 ... keyword args ...
```

```lammps
c_ID = global scalar or vector calculated by a compute with ID
c_ID[I] = Ith component of global vector or Ith column of global array calculated by a compute with ID, I can include wildcard (see below)
f_ID = global scalar or vector calculated by a fix with ID
f_ID[I] = Ith component of global vector or Ith column of global array calculated by a fix with ID, I can include wildcard (see below)
v_name = value(s) calculated by an equal-style or vector-style variable with name
v_name[I] = value calculated by a vector-style variable with name, I can include wildcard (see below)
```

```lammps
*mode* arg = *scalar* or *vector*
  scalar = all input values are global scalars
  vector = all input values are global vectors or global arrays
*ave* args = *one* or *running* or *window M*
  one = output a new average value every Nfreq steps
  running = output cumulative average of all previous Nfreq steps
  window M = output average of M most recent Nfreq steps
*start* args = Nstart
  Nstart = start averaging on this time step
*off* arg = M = do not average this value
  M = value # from 1 to Nvalues
*file* arg = filename
  filename = name of file to output time averages to
*append* arg = filename
  filename = name of file to append time averages to
*overwrite* arg = none = overwrite output file with only latest output
*format* arg = string
  string = C-style format string
*title1* arg = string
  string = text to print as 1st line of output file
*title2* arg = string
  string = text to print as 2nd line of output file
*title3* arg = string
  string = text to print as 3rd line of output file, only for vector mode
```

* ID, group-ID are documented in `fix` command
* ave/time = style name of this fix command
* Nevery = use input values every this many time steps
* Nrepeat = # of times to use input values for calculating averages
* Nfreq = calculate averages every this many time steps
* one or more input values can be listed
* value = c_ID, c_ID[N], f_ID, f_ID[N], v_name
* zero or more keyword/arg pairs may be appended
* keyword = *mode* or *file* or *append* or *ave* or *start* or *off* or *overwrite* or *format* or *title1* or *title2* or *title3*
       *mode* arg = *scalar* or *vector*
       *ave* args = *one* or *running* or *window M*
       *start* args = Nstart

*Examples*

```lammps
fix 1 all ave/time 100 5 1000 c_myTemp c_thermo_temp file temp.profile
fix 1 all ave/time 100 5 1000 c_thermo_press[2] ave window 20 &
                              title1 "My output values"
fix 1 all ave/time 100 5 1000 c_thermo_press[*]
fix 1 all ave/time 1 100 1000 f_indent f_indent[1] file temp.indent off 1
```

Use one or more global values as inputs every few time steps, and
average them over longer timescales.  The resulting averages can be
used by other `output commands` such as
`thermo_style custom`, and can also be written to a
file.  Note that if no time averaging is done, this command can be
used as a convenient way to simply output one or more global values to
a file.

The group specified with this command is ignored.  However, note that
specified values may represent calculations performed by computes and
fixes which store their own "group" definitions.

*Constraints and cautions from this page*

- **NOTE:** For a vector-style variable, only the wildcard forms "\*n" or "m\*n" are allowed.  You must specify the upper bound, because vector-style variable lengths are not determined until the variable is evaluated.  If n is specified larger than the vector length turns out to be, zeroes are output for missing vector values.

#### `fix ave/correlate` — autocorrelation

*Source: `doc/src/fix_ave_correlate.rst`*

```lammps
fix ID group-ID ave/correlate Nevery Nrepeat Nfreq value1 value2 ... keyword args ...
```

```lammps
c_ID = global scalar calculated by a compute with ID
c_ID[I] = Ith component of global vector calculated by a compute with ID, I can include wildcard (see below)
f_ID = global scalar calculated by a fix with ID
f_ID[I] = Ith component of global vector calculated by a fix with ID, I can include wildcard (see below)
v_name = global value calculated by an equal-style variable with name
v_name[I] = Ith component of a vector-style variable with name, I can include wildcard (see below)
```

```lammps
*type* arg = *auto* or *upper* or *lower* or *auto/upper* or *auto/lower* or *full* or *first*
  auto = correlate each value with itself
  upper = correlate each value with each succeeding value
  lower = correlate each value with each preceding value
  auto/upper = auto + upper
  auto/lower = auto + lower
  full = correlate each value with every other value, including itself = auto + upper + lower
  first = correlate each value with the first value
*ave* args = *one* or *running*
  one = zero the correlation accumulation every Nfreq steps
  running = accumulate correlations continuously
*start* args = Nstart
  Nstart = start accumulating correlations on this timestep
*prefactor* args = value
  value = prefactor to scale all the correlation data by
*file* arg = filename
  filename = name of file to output correlation data to
*overwrite* arg = none = overwrite output file with only latest output
*title1* arg = string
  string = text to print as 1st line of output file
*title2* arg = string
  string = text to print as 2nd line of output file
*title3* arg = string
  string = text to print as 3rd line of output file
```

* ID, group-ID are documented in `fix` command
* ave/correlate = style name of this fix command
* Nevery = use input values every this many timesteps
* Nrepeat = # of correlation time windows to accumulate
* Nfreq = calculate time window averages every this many timesteps
* one or more input values can be listed
* value = c_ID, c_ID[N], f_ID, f_ID[N], v_name
* zero or more keyword/arg pairs may be appended
* keyword = *type* or *ave* or *start* or *prefactor* or *file* or *overwrite* or *title1* or *title2* or *title3*
       *type* arg = *auto* or *upper* or *lower* or *auto/upper* or *auto/lower* or *full* or *first*
       *ave* args = *one* or *running*
       *start* args = Nstart

*Examples*

```lammps
fix 1 all ave/correlate 5 100 1000 c_myTemp file temp.correlate
fix 1 all ave/correlate 1 50 10000 &
          c_thermo_press[1] c_thermo_press[2] c_thermo_press[3] &
          type upper ave running title1 "My correlation data"
fix 1 all ave/correlate 1 50 10000 c_thermo_press[*]
```

Use one or more global scalar values as inputs every few timesteps,
calculate time correlations between them at varying time intervals,
and average the correlation data over longer timescales.  The resulting
correlation values can be time integrated by
`variables` or used by other
`output commands` such as
`thermo_style custom`, and can also be written to a file.
See the `fix ave/correlate/long` command for an
alternate method for computing correlation functions efficiently over
very long time windows.

The group specified with this command is ignored.  However, note that
specified values may represent calculations performed by computes and
fixes which store their own "group" definitions.

*Constraints and cautions from this page*

- **NOTE:** For a vector-style variable, only the wildcard forms "\*n" or "m\*n" are allowed.  You must specify the upper bound, because vector-style variable lengths are not determined until the variable is evaluated.  If n is specified larger than the vector length turns out to be, zeroes are output for missing vector values.

---

## 2. Elastic constants and cell relaxation

*Why this is here:* Decides how strain SHARES between two coherently bonded materials, and therefore the relaxed in-plane constant of an interface. Measured here: SiCGe.tersoff reproduces a0 to 0.02 % but gets C_ij 8-18 % wrong and INVERTS which of Si/Ge is biaxially stiffer, so this section is a validation gate, not background reading.

### Official scripts

#### `ELASTIC/in.elastic` — Full elastic tensor by finite differences

```lammps
# Compute elastic constant tensor for a crystal
#
# Written by Aidan Thompson (Sandia, athomps@sandia.gov)
#
#  This script uses the following three include files.
#
#   init.mod      (must be modified for different crystal structures)
# 	       	  Define units, deformation parameters and initial
#		  configuration of the atoms and simulation cell.  
#
#
#   potential.mod    (must be modified for different pair styles)
# 		     Define pair style and other attributes 
#		     not stored in restart file
#
#
#   displace.mod    (displace.mod should not need to be modified)
# 		    Perform positive and negative box displacements 
# 		    in direction ${dir} and size ${up}. 
# 		    It uses the resultant changes 
#		    in stress to compute one
# 		    row of the elastic stiffness tensor
#		    
#		    Inputs variables:
#		    	   dir = the Voigt deformation component 
#		    		    (1,2,3,4,5,6)  
#		    Global constants:
#       	    	   up = the deformation magnitude (strain units)
#       		   cfac = conversion from LAMMPS pressure units to 
#               	   output units for elastic constants 
#
#
#  To run this on a different system, it should only be necessary to 
#  modify the files init.mod and potential.mod. In order to calculate
#  the elastic constants correctly, care must be taken to specify
#  the correct units in init.mod (units, cfac and cunits). It is also
#  important to verify that the minimization of energy w.r.t atom
#  positions in the deformed cell is fully converged.
#  One indication of this is that the elastic constants are insensitive
#  to the choice of the variable ${up} in init.mod. Another is to check
#  the final max and two-norm forces reported in the log file. If you know
#  that minimization is not required, you can set maxiter = 0.0 in 
#  init.mod. 
#

include init.mod
include potential.mod

# Compute initial state
fix 3 all box/relax  aniso 0.0
minimize ${etol} ${ftol} ${maxiter} ${maxeval}

variable tmp equal pxx
variable pxx0 equal ${tmp}
variable tmp equal pyy
variable pyy0 equal ${tmp}
variable tmp equal pzz
variable pzz0 equal ${tmp}
variable tmp equal pyz
variable pyz0 equal ${tmp}
variable tmp equal pxz
variable pxz0 equal ${tmp}
variable tmp equal pxy
variable pxy0 equal ${tmp}

variable tmp equal lx
variable lx0 equal ${tmp}
variable tmp equal ly
variable ly0 equal ${tmp}
variable tmp equal lz
variable lz0 equal ${tmp}

# These formulas define the derivatives w.r.t. strain components
# Constants uses $, variables use v_ 
variable d1 equal -(v_pxx1-${pxx0})/(v_delta/v_len0)*${cfac}
variable d2 equal -(v_pyy1-${pyy0})/(v_delta/v_len0)*${cfac}
variable d3 equal -(v_pzz1-${pzz0})/(v_delta/v_len0)*${cfac}
variable d4 equal -(v_pyz1-${pyz0})/(v_delta/v_len0)*${cfac}
variable d5 equal -(v_pxz1-${pxz0})/(v_delta/v_len0)*${cfac}
variable d6 equal -(v_pxy1-${pxy0})/(v_delta/v_len0)*${cfac}

displace_atoms all random ${atomjiggle} ${atomjiggle} ${atomjiggle} 87287 units box

# Write restart
unfix 3
write_restart restart.equil

# uxx Perturbation

variable dir equal 1
include displace.mod

# uyy Perturbation

variable dir equal 2
include displace.mod

# uzz Perturbation

variable dir equal 3
include displace.mod

# uyz Perturbation

variable dir equal 4
include displace.mod

# uxz Perturbation

variable dir equal 5
include displace.mod

# uxy Perturbation

variable dir equal 6
include displace.mod

# Output final values

variable C11all equal ${C11}
variable C22all equal ${C22}
variable C33all equal ${C33}

variable C12all equal 0.5*(${C12}+${C21})
variable C13all equal 0.5*(${C13}+${C31})
variable C23all equal 0.5*(${C23}+${C32})

variable C44all equal ${C44}
variable C55all equal ${C55}
variable C66all equal ${C66}

variable C14all equal 0.5*(${C14}+${C41})
variable C15all equal 0.5*(${C15}+${C51})
variable C16all equal 0.5*(${C16}+${C61})

variable C24all equal 0.5*(${C24}+${C42})
variable C25all equal 0.5*(${C25}+${C52})
variable C26all equal 0.5*(${C26}+${C62})

variable C34all equal 0.5*(${C34}+${C43})
# ... truncated at 140 lines; see examples/ELASTIC/in.elastic
```

#### `ELASTIC/init.mod` — Crystal, units and strain magnitude

```lammps
# NOTE: This script can be modified for different atomic structures, 
# units, etc. See in.elastic for more info.
#

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
variable up equal 1.0e-6
 
# Define the amount of random jiggle for atoms
# This prevents atoms from staying on saddle points
variable atomjiggle equal 1.0e-5

# Uncomment one of these blocks, depending on what units
# you are using in LAMMPS and for output

# metal units, elastic constants in eV/A^3
#units		metal
#variable cfac equal 6.2414e-7
#variable cunits string eV/A^3

# metal units, elastic constants in GPa
units		metal
variable cfac equal 1.0e-4
variable cunits string GPa

# real units, elastic constants in GPa
#units		real
#variable cfac equal 1.01325e-4
#variable cunits string GPa

# Define minimization parameters
variable etol equal 0.0 
variable ftol equal 1.0e-10
variable maxiter equal 100
variable maxeval equal 1000
variable dmax equal 1.0e-2

# generate the box and atom positions using a diamond lattice
variable a equal 5.43

boundary	p p p

lattice         diamond $a
region		box prism 0 2.0 0 3.0 0 4.0 0.0 0.0 0.0
create_box	1 box
create_atoms	1 box

# Need to set mass to something, just to satisfy LAMMPS
mass 1 1.0e-20
```

#### `ELASTIC/potential.mod` — Pair style and minimisation settings

```lammps
# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Choose potential
pair_style	sw
pair_coeff * * Si.sw Si

# Setup neighbor style
neighbor 1.0 nsq
neigh_modify once no every 1 delay 0 check yes

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		1
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no
```

#### `ELASTIC/displace.mod` — Applies +/- strain and reads the stress

```lammps
# NOTE: This script should not need to be
# modified. See in.elastic for more info.
#
# Find which reference length to use

if "${dir} == 1" then &
   "variable len0 equal ${lx0}" 
if "${dir} == 2" then &
   "variable len0 equal ${ly0}" 
if "${dir} == 3" then &
   "variable len0 equal ${lz0}" 
if "${dir} == 4" then &
   "variable len0 equal ${lz0}" 
if "${dir} == 5" then &
   "variable len0 equal ${lz0}" 
if "${dir} == 6" then &
   "variable len0 equal ${ly0}" 

# Reset box and simulation parameters

clear
box tilt large
read_restart restart.equil
include potential.mod

# Negative deformation

variable delta equal -${up}*${len0}
variable deltaxy equal -${up}*xy
variable deltaxz equal -${up}*xz
variable deltayz equal -${up}*yz
if "${dir} == 1" then &
   "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"
if "${dir} == 2" then &
   "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"
if "${dir} == 3" then &
   "change_box all z delta 0 ${delta} remap units box"
if "${dir} == 4" then &
   "change_box all yz delta ${delta} remap units box"
if "${dir} == 5" then &
   "change_box all xz delta ${delta} remap units box"
if "${dir} == 6" then &
   "change_box all xy delta ${delta} remap units box"

# Relax atoms positions

minimize ${etol} ${ftol} ${maxiter} ${maxeval}

# Obtain new stress tensor
 
variable tmp equal pxx
variable pxx1 equal ${tmp}
variable tmp equal pyy
variable pyy1 equal ${tmp}
variable tmp equal pzz
variable pzz1 equal ${tmp}
variable tmp equal pxy
variable pxy1 equal ${tmp}
variable tmp equal pxz
variable pxz1 equal ${tmp}
variable tmp equal pyz
variable pyz1 equal ${tmp}

# Compute elastic constant from pressure tensor

variable C1neg equal ${d1}
variable C2neg equal ${d2}
variable C3neg equal ${d3}
variable C4neg equal ${d4}
variable C5neg equal ${d5}
variable C6neg equal ${d6}

# Reset box and simulation parameters

clear
box tilt large
read_restart restart.equil
include potential.mod

# Positive deformation

variable delta equal ${up}*${len0}
variable deltaxy equal ${up}*xy
variable deltaxz equal ${up}*xz
variable deltayz equal ${up}*yz
if "${dir} == 1" then &
   "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"
if "${dir} == 2" then &
   "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"
if "${dir} == 3" then &
   "change_box all z delta 0 ${delta} remap units box"
if "${dir} == 4" then &
   "change_box all yz delta ${delta} remap units box"
if "${dir} == 5" then &
   "change_box all xz delta ${delta} remap units box"
if "${dir} == 6" then &
   "change_box all xy delta ${delta} remap units box"

# Relax atoms positions

minimize ${etol} ${ftol} ${maxiter} ${maxeval}

# Obtain new stress tensor
 
variable tmp equal pe
variable e1 equal ${tmp}
variable tmp equal press
variable p1 equal ${tmp}
variable tmp equal pxx
variable pxx1 equal ${tmp}
variable tmp equal pyy
variable pyy1 equal ${tmp}
variable tmp equal pzz
variable pzz1 equal ${tmp}
variable tmp equal pxy
variable pxy1 equal ${tmp}
variable tmp equal pxz
variable pxz1 equal ${tmp}
variable tmp equal pyz
variable pyz1 equal ${tmp}

# Compute elastic constant from pressure tensor

variable C1pos equal ${d1}
variable C2pos equal ${d2}
variable C3pos equal ${d3}
variable C4pos equal ${d4}
variable C5pos equal ${d5}
variable C6pos equal ${d6}

# Combine positive and negative 

variable C1${dir} equal 0.5*(${C1neg}+${C1pos})
variable C2${dir} equal 0.5*(${C2neg}+${C2pos})
variable C3${dir} equal 0.5*(${C3neg}+${C3pos})
variable C4${dir} equal 0.5*(${C4neg}+${C4pos})
variable C5${dir} equal 0.5*(${C5neg}+${C5pos})
variable C6${dir} equal 0.5*(${C6neg}+${C6pos})

# Delete dir to make sure it is not reused
# ... truncated at 140 lines; see examples/ELASTIC/displace.mod
```

### Command reference

#### `fix box/relax` — relax the cell to zero stress

*Source: `doc/src/fix_box_relax.rst`*

```lammps
fix ID group-ID box/relax keyword value ...
```

```lammps
one or more keyword value pairs may be appended
keyword = *iso* or *aniso* or *tri* or *x* or *y* or *z* or *xy* or *yz* or *xz* or *couple* or *nreset* or *vmax* or *dilate* or *scaleyz* or *scalexz* or *scalexy* or *fixedpoint*
  *iso* or *aniso* or *tri* value = Ptarget = desired pressure (pressure units)
  *x* or *y* or *z* or *xy* or *yz* or *xz* value = Ptarget = desired pressure (pressure units)
  *couple* = *none* or *xyz* or *xy* or *yz* or *xz*
  *nreset* value = reset reference cell every this many minimizer iterations
  *vmax* value = fraction = max allowed volume change in one iteration
  *dilate* value = *all* or *partial*
  *scaleyz* value = *yes* or *no* = scale yz with lz
  *scalexz* value = *yes* or *no* = scale xz with lz
  *scalexy* value = *yes* or *no* = scale xy with ly
  *fixedpoint* values = x y z
    x,y,z = perform relaxation dilation/contraction around this point (distance units)
```

* ID, group-ID are documented in `fix` command
* box/relax = style name of this fix command
       *iso* or *aniso* or *tri* value = Ptarget = desired pressure (pressure units)
       *x* or *y* or *z* or *xy* or *yz* or *xz* value = Ptarget = desired pressure (pressure units)
       *couple* = *none* or *xyz* or *xy* or *yz* or *xz*
       *nreset* value = reset reference cell every this many minimizer iterations
       *vmax* value = fraction = max allowed volume change in one iteration
       *dilate* value = *all* or *partial*
       *scaleyz* value = *yes* or *no* = scale yz with lz
       *scalexz* value = *yes* or *no* = scale xz with lz
       *scalexy* value = *yes* or *no* = scale xy with ly
       *fixedpoint* values = x y z

*Examples*

```lammps
fix 1 all box/relax iso 0.0 vmax 0.001
fix 2 water box/relax aniso 0.0 dilate partial
fix 2 ice box/relax tri 0.0 couple xy nreset 100
```

Apply an external pressure or stress tensor to the simulation box
during an `energy minimization`.  This allows the box
size and shape to vary during the iterations of the minimizer so that
the final configuration will be both an energy minimum for the
potential energy of the atoms, and the system pressure tensor will be
close to the specified external tensor.  Conceptually, specifying a
positive pressure is like squeezing on the simulation box; a negative
pressure typically allows the box to expand.

The external pressure tensor is specified using one or more of the
*iso*, *aniso*, *tri*, *x*, *y*, *z*, *xy*, *xz*, *yz*, and *couple*
keywords.  These keywords give you the ability to specify all 6
components of an external stress tensor, and to couple various of
these components together so that the dimensions they represent are
varied together during the minimization.

*Constraints and cautions from this page*

- **NOTE:** Applying an external pressure to tilt dimensions *xy*, *xz*, *yz* can sometimes result in arbitrarily large values of the tilt factors, i.e. a dramatically deformed simulation box.  This typically indicates that there is something badly wrong with how the simulation was constructed.  The two most common sources of this error are applying a shear stress to a liquid system or specifying an external shear stress tensor that exceeds the yield stress of the solid.  In either case the minimization may converge to a bogus conformation or not converge at all.  Also note that if the box shape tilts to an extreme shape, LAMMPS will run less efficiently, due to the large volume of communication needed to acquire ghost atoms around a processor's irregular-shaped subdomain.  For extreme values of tilt, LAMMPS may also lose atoms and generate an error.
- **NOTE:** Performing a minimization with this fix is not a mathematically well-defined minimization problem.  This is because the objective function being minimized changes if the box size/shape changes.  In practice this means the minimizer can get "stuck" before you have reached the desired tolerance.  The solution to this is to restart the minimizer from the new adjusted box size/shape, since that creates a new objective function valid for the new box size/shape.  Repeat as necessary until the box size/shape has reached its new equilibrium.
- **NOTE:** As normally computed, pressure includes a kinetic- energy or temperature-dependent component; see the `compute pressure` command.  However, atom velocities are ignored during a minimization, and the applied pressure(s) specified with this command are assumed to only be the virial component of the pressure (the non-kinetic portion).  Thus if atoms have a non-zero temperature and you print the usual thermodynamic pressure, it may not appear the system is converging to your specified pressure.  The solution for this is to either (a) zero the velocities of all atoms before performing the minimization, or (b) make sure you are monitoring the pressure without its kinetic component.  The latter can be done by outputting the pressure from the pressure compute this command creates (see below) or a pressure compute you define yourself.
- **NOTE:** Because pressure is often a very sensitive function of volume, it can be difficult for the minimizer to equilibrate the system the desired pressure with high precision, particularly for solids.  Some techniques that seem to help are (a) use the "min_modify line quadratic" option when minimizing with box relaxations, (b) minimize several times in succession if need be, to drive the pressure closer to the target pressure, (c) relax the atom positions before relaxing the box, and (d) relax the box to the target hydrostatic pressure before relaxing to a target shear stress state. Also note that some systems (e.g. liquids) will not sustain a non-hydrostatic applied pressure, which means the minimizer will not converge.
- **NOTE:** If both the *temp* and *press* keywords are used in a single thermo_modify command (or in two separate commands), then the order in which the keywords are specified is important.  Note that a `pressure compute` defines its own temperature compute as an argument when it is specified.  The *temp* keyword will override this (for the pressure compute being used by fix box/relax), but only if the *temp* keyword comes after the *press* keyword.  If the *temp* keyword comes before the *press* keyword, then the new pressure compute specified by the *press* keyword will be unaffected by the *temp* setting.

#### `minimize` — energy minimisation

*Source: `doc/src/minimize.rst`*

```lammps
minimize etol ftol maxiter maxeval
```

* etol = stopping tolerance for energy (unitless)
* ftol = stopping tolerance for force (force units)
* maxiter = max iterations of minimizer
* maxeval = max number of force/energy evaluations

*Examples*

```lammps
minimize 1.0e-4 1.0e-6 100 1000
minimize 0.0 1.0e-8 1000 100000
```

Perform an energy minimization of the system, by iteratively adjusting
atom coordinates.  Iterations are terminated when one of the stopping
criteria is satisfied.  At that point the configuration will hopefully
be in a local potential energy minimum.  More precisely, the
configuration should approximate a critical point for the objective
function (see below), which may or may not be a local minimum.

The minimization algorithm used is set by the `min_style` command.  Other options are set by the `min_modify` command.  Minimize commands can be interspersed with
`run` commands to alternate between relaxation and dynamics.
The minimizers bound the distance atoms may move in one iteration, so
that you can relax systems with highly overlapped atoms (large energies
and forces) by pushing the atoms off of each other.

*Constraints and cautions from this page*

- **NOTE:** The damped dynamic minimizer algorithms will use the timestep you have defined via the `timestep` command or its default value.  Often they will converge more quickly if you use a timestep about 10x larger than you would normally use for regular molecular dynamics simulations.
- **NOTE:** the `minimization style` *spin*, *spin/cg*, and *spin/lbfgs* replace the force tolerance *ftol* by a torque tolerance. The minimization procedure stops if the 2-norm (length) of the torque vector on atom (defined as the cross product between the atomic spin and its precession vectors omega) is less than *ftol*, or if any of the other criteria are met. Torque have the same units as the energy.
- **NOTE:** You can also use the `fix halt` command to specify a general criterion for exiting a minimization, that is a calculation performed on the state of the current system, as defined by an `equal-style variable`.
- **NOTE:** There are several force fields in LAMMPS which have discontinuities or other approximations which may prevent you from performing an energy minimization to tight tolerances.  For example, you should use a `pair style` that goes to 0.0 at the cutoff distance when performing minimization (even if you later change it when running dynamics).  If you do not do this, the total energy of the system will have discontinuities when the relative distance between any pair of atoms changes from cutoff *plus* epsilon to cutoff *minus* epsilon and the minimizer may thus behave poorly. Some of the many-body potentials use splines and other internal cutoffs that inherently have this problem.  The `long-range Coulombic styles` (PPPM, Ewald) are approximate to within the user-specified tolerance, which means their energy and forces may not agree to a higher precision than the Kspace-specified tolerance.  This agreement is further reduced when using tabulation to speed up the computation of the real-space part of the Coulomb interactions, which is enabled by default.  In all these cases, the minimizer may give up and stop before finding a minimum to the specified energy or force tolerance.
- **NOTE:** Some fixes which are invoked during minimization have an associated potential energy.  For that energy to be included in the total potential energy of the system (the quantity being minimized), you MUST enable the `fix_modify` *energy* option for that fix.  The doc pages for individual `fix` commands specify if this should be done.

#### `change_box` — impose a deformation

*Source: `doc/src/change_box.rst`*

```lammps
change_box group-ID parameter args ... keyword args ...
```

```lammps
parameter = *x* or *y* or *z* or *xy* or *xz* or *yz* or *boundary* or *ortho* or *triclinic* or *set* or *remap*
  *x*, *y*, *z* args = style value(s)
    style = *final* or *delta* or *scale* or *volume*
      *final* values = lo hi
        lo hi = box boundaries after displacement (distance units)
      *delta* values = dlo dhi
        dlo dhi = change in box boundaries after displacement (distance units)
      *scale* values = factor
        factor = multiplicative factor for change in box length after displacement
      *volume* value = none = adjust this dim to preserve volume of system
  *xy*, *xz*, *yz* args = style value
    style = *final* or *delta*
      *final* value = tilt
        tilt = tilt factor after displacement (distance units)
      *delta* value = dtilt
        dtilt = change in tilt factor after displacement (distance units)
  *boundary* args = x y z
    x,y,z = *p* or *s* or *f* or *m*, one or two letters
    *p* is periodic
    *f* is non-periodic and fixed
    *s* is non-periodic and shrink-wrapped
    *m* is non-periodic and shrink-wrapped with a minimum value
  *ortho* args = none = change box to orthogonal
  *triclinic* args = none = change box to triclinic
  *set* args = none = store state of current box
  *remap* args = none = remap atom coords from last saved state to current box
```

```lammps
*units* value = *lattice* or *box*
  lattice = distances are defined in lattice units
  box = distances are defined in simulation box units
```

* group-ID = ID of group of atoms to (optionally) displace
* one or more parameter/arg pairs may be appended
       *x*, *y*, *z* args = style value(s)
           *final* values = lo hi
           *delta* values = dlo dhi
           *scale* values = factor
           *volume* value = none = adjust this dim to preserve volume of system
       *xy*, *xz*, *yz* args = style value
           *final* value = tilt
           *delta* value = dtilt
       *boundary* args = x y z
         *p* is periodic

*Examples*

```lammps
change_box all xy final -2.0 z final 0.0 5.0 boundary p p f remap units box
change_box all x scale 1.1 y volume z volume remap
```

Change the volume and/or shape and/or boundary conditions for the
simulation box.  Orthogonal simulation boxes have 3 adjustable size
parameters (x,y,z).  Triclinic (non-orthogonal) simulation boxes have
6 adjustable size/shape parameters (x,y,z,xy,xz,yz).  Any or all of
them can be adjusted independently by this command.  Thus it can be
used to expand or contract a box, or to apply a shear strain to a
non-orthogonal box.  It can also be used to change the boundary
conditions for the simulation box, similar to the
`boundary` command.

The size and shape of the initial simulation box are specified by the
`create_box` or `read_data` or
`read_restart` command used to setup the simulation.
The size and shape may be altered by subsequent runs, e.g. by use of
the `fix npt` or `fix deform` commands.
The `create_box`, `read data`, and
`read_restart` commands also determine whether the
simulation box is orthogonal or triclinic and their doc pages explain
the meaning of the xy,xz,yz tilt factors.

*Constraints and cautions from this page*

- **NOTE:** This means that you cannot use the change_box command to enlarge a shrink-wrapped box, e.g. to make room to insert more atoms via the `create_atoms` command, because the simulation box will be re-shrink-wrapped before the change_box command completes. Instead you could do something like this, assuming the simulation box is non-periodic and atoms extend from 0 to 20 in all dimensions:
- **NOTE:** Unlike the earlier "displace_box" version of this command, atom remapping is NOT performed by default.  This command allows remapping to be done in a more general way, exactly when you specify it (zero or more times) in the sequence of transformations.  Thus if you do not use the *remap* keyword, atom coordinates will not be changed even if the box size/shape changes.  If a uniformly strained state is desired, the *remap* keyword should be specified.
- **NOTE:** It is possible to lose atoms with this command.  E.g. by changing the box without remapping the atoms, and having atoms end up outside of non-periodic boundaries.  It is also possible to alter bonds between atoms straddling a boundary in bad ways.  E.g. by converting a boundary from periodic to non-periodic.  It is also possible when remapping atoms to put them (nearly) on top of each other.  E.g. by converting a boundary from non-periodic to periodic. All of these will typically lead to bad dynamics and/or generate error messages.
- **NOTE:** The simulation box size/shape can be changed by arbitrarily large amounts by this command.  This is not a problem, except that the mapping of processors to the simulation box is not changed from its initial 3d configuration; see the `processors` command.  Thus, if the box size/shape changes dramatically, the mapping of processors to the simulation box may not end up as optimal as the initial mapping attempted to be.  You may wish to re-balance the atoms by using the `balance` command if that is the case.
- **NOTE:** You cannot use this command after reading a restart file (and before a run is performed) if the restart file stored per-atom information from a fix and any of the specified keywords change the box size or shape or boundary conditions.  This is because atoms may be moved to new processors and the restart info will not migrate with them.  LAMMPS will generate an error if this could happen.  Only the *ortho* and *triclinic* keywords do not trigger this error.  One solution is to perform a "run 0" command before using the change_box command.  This clears the per-atom restart data, whether it has been re-assigned to a new fix or not.
- **NOTE:** Because the keywords used in this command are applied one at a time to the simulation box and the atoms in it, care must be taken with triclinic cells to avoid exceeding the limits on skew after each transformation in the sequence.  If skew is exceeded before the final transformation this can be avoided by changing the order of the sequence, or breaking the transformation into two or more smaller transformations.  For more information on the allowed limits for box skew see the discussion on triclinic boxes on `Howto triclinic` doc page.

#### `compute born/matrix` — elastic constants without finite differences

*Source: `doc/src/compute_born_matrix.rst`*

```lammps
compute ID group-ID born/matrix keyword value ...
```

```lammps
keyword = *numdiff* or *pair* or *bond* or *angle* or *dihedral* or *improper*
  *numdiff* values = delta virial-ID
    delta = magnitude of strain (dimensionless)
    virial-ID = ID of pressure compute for virial (string)
    (*numdiff* cannot be used with any other keyword)
  *pair* = compute pair-wise contributions
  *bond* = compute bonding contributions
  *angle* = compute angle contributions
  *dihedral* = compute dihedral contributions
  *improper* = compute improper contributions
```

* ID, group-ID are documented in `compute` command
* born/matrix = style name of this compute command
* zero or more keywords or keyword/value pairs may be appended
       *numdiff* values = delta virial-ID
       *pair* = compute pair-wise contributions
       *bond* = compute bonding contributions
       *angle* = compute angle contributions
       *dihedral* = compute dihedral contributions
       *improper* = compute improper contributions

*Examples*

```lammps
compute 1 all born/matrix
compute 1 all born/matrix bond angle
compute 1 all born/matrix numdiff 1.0e-4 myvirial
```

Define a compute that calculates
$\frac{\partial{}^2U}{\partial\varepsilon_{i}\partial\varepsilon_{j}},$ the
second derivatives of the potential energy $U$ with respect to the strain
tensor $\varepsilon$ elements. These values are related to:

C^{B}_{i,j}=\frac{1}{V}\frac{\partial{}^2U}{\partial{}\varepsilon_{i}\partial\varepsilon_{j}}

*Constraints and cautions from this page*

- **NOTE:** It is important to choose a suitable value for delta, the magnitude of strains that are used to generate finite difference approximations to the exact virial stress.  For typical systems, a value in the range of 1 part in 1e5 to 1e6 will be sufficient. However, the best value will depend on a multitude of factors including the stiffness of the interatomic potential, the thermodynamic state of the material being probed, and so on. The only way to be sure that you have made a good choice is to do a sensitivity study on a representative atomic configuration, sweeping over a wide range of values of delta. If delta is too small, the output values will vary erratically due to truncation effects. If delta is increased beyond a certain point, the output values will start to vary smoothly with delta, due to growing contributions from higher order derivatives. In between these two limits, the numerical virial values should be largely independent of delta.

#### `fix numdiff` — numerical force derivatives

*Source: `doc/src/fix_numdiff.rst`*

```lammps
fix ID group-ID numdiff Nevery delta
```

* ID, group-ID are documented in `fix` command
* numdiff = style name of this fix command
* Nevery = calculate force by finite difference every this many timesteps
* delta = size of atom displacements (distance units)

*Examples*

```lammps
fix 1 all numdiff 10 1e-6
fix 1 movegroup numdiff 100 0.01
```

Calculate forces through finite difference calculations of energy
versus position.  These forces can be compared to analytic forces
computed by pair styles, bond styles, etc.  This can be useful for
debugging or other purposes.

The group specified with the command means only atoms within the group
have their averages computed.  Results are set to 0.0 for atoms not in
the group.

*Constraints and cautions from this page*

- **NOTE:** It is important to choose a suitable value for delta, the magnitude of atom displacements that are used to generate finite difference approximations to the exact forces.  For typical systems, a value in the range of 1 part in 1e4 to 1e5 of the typical separation distance between atoms in the liquid or solid state will be sufficient. However, the best value will depend on a multitude of factors including the stiffness of the interatomic potential, the thermodynamic state of the material being probed, and so on. The only way to be sure that you have made a good choice is to do a sensitivity study on a representative atomic configuration, sweeping over a wide range of values of delta.  If delta is too small, the output forces will vary erratically due to truncation effects. If delta is increased beyond a certain point, the output forces will start to vary smoothly with delta, due to growing contributions from higher order derivatives. In between these two limits, the numerical force values should be largely independent of delta.
- **NOTE:** The cost of each energy evaluation is essentially the cost of an MD timestep.  Thus invoking this fix once for a 3d system has a cost of 6N timesteps, where N is the total number of atoms in the system. So this fix can be very expensive to use for large systems. One expedient alternative is to define the fix for a group containing only a few atoms.

#### `compute stress/atom` — per-atom stress

*Source: `doc/src/compute_stress_atom.rst`*

```lammps
compute ID group-ID style temp-ID keyword ...
```

* ID, group-ID are documented in `compute` command
* style = *stress/atom* or *centroid/stress/atom*
* temp-ID = ID of compute that calculates temperature, can be NULL if not needed
* zero or more keywords may be appended
* keyword = *ke* or *pair* or *bond* or *angle* or *dihedral* or *improper* or *kspace* or *fix* or *virial*

*Examples*

```lammps
compute 1 mobile stress/atom NULL
compute 1 mobile stress/atom myRamp
compute 1 all stress/atom NULL pair bond
compute 1 all centroid/stress/atom NULL bond dihedral improper
```

Define a computation that computes per-atom stress tensor for each
atom in a group.  In case of compute *stress/atom*, the tensor for
each atom is symmetric with 6 components and is stored as a 6-element
vector in the following order: $xx$, $yy$, $zz$,
$xy$, $xz$, $yz$.  In case of compute
*centroid/stress/atom*, the tensor for each atom is asymmetric with 9
components and is stored as a 9-element vector in the following order:
$xx$, $yy$, $zz$, $xy$, $xz$,
$yz$, $yx$, $zx$, $zy$.  See the `compute
pressure` command if you want the stress tensor
(pressure) of the entire system.

The stress tensor for atom $I$ is given by the following
formula, where $a$ and $b$ take on values $x$,
$y$, $z$ to generate the components of the tensor:

*Constraints and cautions from this page*

- **NOTE:** The per-atom stress does not include any Lennard-Jones tail corrections to the pressure added by the `pair_modify tail yes` command, since those are contributions to the global system pressure.

#### `compute pressure` — pressure tensor

*Source: `doc/src/compute_pressure.rst`*

```lammps
compute ID group-ID pressure temp-ID keyword ...
```

* ID, group-ID are documented in `compute` command
* pressure = style name of this compute command
* temp-ID = ID of compute that calculates temperature, can be NULL if not needed
* zero or more keywords may be appended
* keyword = *ke* or *pair* or *bond* or *angle* or *dihedral* or *improper* or *kspace* or *fix* or *virial* or *pair/hybrid*

*Examples*

```lammps
compute 1 all pressure thermo_temp
compute 1 all pressure NULL pair bond
compute 1 all pressure NULL pair/hybrid lj/cut
```

Define a computation that calculates the pressure of the entire system
of atoms.  The specified group must be "all".  See the
`compute stress/atom` command if you want per-atom
pressure (stress).  These per-atom values could be summed for a group
of atoms via the `compute reduce` command.

The pressure is computed by the formula

---

## 3. Hardness and indentation

*Why this is here:* Contact microhardness H_c is the single largest lever on the contact-conductance result (h_c depends on P/H_c), and the Song-Yovanovich correlation needs load-dependent indentation data. fix indent is the atomistic route to it.

### Official scripts

#### `indent/in.indent` — Indenter pushed into a 2d solid

```lammps
# 2d indenter simulation

dimension       2
boundary        p s p

atom_style      atomic
neighbor        0.3 bin
neigh_modify    delay 5

# create geometry

lattice         hex 0.9
region          box block 0 20 0 10 -0.25 0.25
create_box      2 box
create_atoms    1 box

mass            1 1.0
mass            2 1.0

# LJ potentials

pair_style      lj/cut 2.5
pair_coeff      * * 1.0 1.0 2.5

# define groups

region          1 block INF INF INF 1.25 INF INF
group           lower region 1
group           mobile subtract all lower
set             group lower type 2

# initial velocities

compute         new mobile temp
velocity        mobile create 0.2 482748 temp new
fix             1 all nve
fix             2 lower setforce 0.0 0.0 0.0
fix             3 all temp/rescale 100 0.1 0.1 0.01 1.0

# run with indenter

timestep        0.003
variable        k equal 1000.0/xlat
variable        y equal "13.0*ylat - step*dt*0.02*ylat"

fix             4 all indent $k sphere 10 v_y 0 5.0
fix             5 all enforce2d

thermo          1000
thermo_modify   temp new

#dump           1 all atom 250 dump.indent

#dump           2 all image 1000 image.*.jpg type type &
#               zoom 1.6 adiam 1.5 fix 4 type 0.0 -2.0
#dump_modify    2 pad 5 fcolor purple

#dump           3 all movie 1000 movie.mpg type type &
#               zoom 1.6 adiam 1.5
#dump_modify    3 pad 5

run             30000

# run without indenter

unfix           4
#undump          2
#dump           2 all image 1000 image.*.jpg type type &
#               zoom 1.6 adiam 1.5
#dump_modify    2 pad 5
run             30000
```

#### `indent/in.indent.min` — Indentation by minimisation

```lammps
# 2d indenter simulation with minimization instead of dynamics

dimension     2
boundary      p s p

atom_style    atomic
neighbor      0.3 bin
neigh_modify  delay 5

# create geometry

lattice	      hex 0.9
region	      box block 0 20 0 10 -0.25 0.25
create_box    2 box
create_atoms  1 box

mass	      * 1.0

# LJ potentials

pair_style    lj/cut 2.5
pair_coeff    * * 1.0 1.0 2.5

# define groups

region        1 block INF INF INF 1.25 INF INF
group	      lower region 1
group	      mobile subtract all lower
set	      group lower type 2

# initial velocities

fix	      2 lower setforce 0.0 0.0 0.0

# indenter

fix	      5 all enforce2d

# minimize with indenter

thermo	      10

#dump	      1 all atom 10 dump.indent

#dump	      2 all image 25 image.*.jpg type type &
#	      zoom 1.6 adiam 1.5
#dump_modify   2 pad 4

#dump	      3 all movie 25 movie.mpg type type &
#	      zoom 1.6 adiam 1.5
#dump_modify   3 pad 4

minimize      1.0e-6 1.0e-6 1000 1000

variable      k equal 5000.0/xlat
variable      k1 equal 1000.0/xlat

fix           4 all indent $k sphere 10 13.0 0 6.0
fix_modify    4 energy yes
minimize      1.0e-6 1.0e-6 1000 1000

fix           4 all indent ${k1} sphere 10 12.5 0 6.0
fix_modify    4 energy yes
minimize      1.0e-6 1.0e-6 1000 1000

fix           4 all indent ${k1} sphere 10 12.0 0 6.0
fix_modify    4 energy yes
minimize      1.0e-6 1.0e-6 1000 1000

fix           4 all indent ${k1} sphere 10 11.4 0 6.0
fix_modify    4 energy yes
minimize      1.0e-6 1.0e-6 1000 1000

fix           4 all indent ${k1} sphere 10 11.2 0 6.0
fix_modify    4 energy yes
minimize      1.0e-6 1.0e-6 1000 1000

fix	      4 all indent ${k1} sphere 10 11.0 0 6.0
fix_modify    4 energy yes
minimize      1.0e-6 1.0e-6 1000 1000
```

### Command reference

#### `fix indent` — spherical or cylindrical indenter

*Source: `doc/src/fix_indent.rst`*

```lammps
fix ID group-ID indent K gstyle args keyword value ...
```

```lammps
*sphere* args = x y z R
  x, y, z = position of center of indenter (distance units)
  R = sphere radius of indenter (distance units)
  any of x, y, z, R can be a variable (see below)
*cylinder* args = dim c1 c2 R
  dim = *x* or *y* or *z* = axis of cylinder
  c1, c2 = coords of cylinder axis in other 2 dimensions (distance units)
  R = cylinder radius of indenter (distance units)
  any of c1,c2,R can be a variable (see below)
*cone* args = dim c1 c2 radlo radhi lo hi
  dim = *x* or *y* or *z* = axis of cone
  c1, c2 = coords of cone axis in other 2 dimensions (distance units)
  radlo,radhi = cone radii at lo and hi end (distance units)
  lo,hi = bounds of cone in dim (distance units)
  any of c1, c2, radlo, radhi, lo, hi can be a variable (see below)
*plane* args = dim pos side
  dim = *x* or *y* or *z* = plane perpendicular to this dimension
  pos = position of plane in dimension x, y, or z (distance units)
  pos can be a variable (see below)
  side = *lo* or *hi*
```

```lammps
*side* value = *in* or *out*
  *in* = the indenter acts on particles inside the sphere or cylinder or cone
  *out* = the indenter acts on particles outside the sphere or cylinder or cone
*units* value = *lattice* or *box*
  lattice = the geometry is defined in lattice units
  box = the geometry is defined in simulation box units
```

* ID, group-ID are documented in `fix` command
* indent = style name of this fix command
* K = force constant for indenter surface (force/distance\^2 units)
* gstyle = *sphere* or *cylinder* or *cone* or *plane*
       *sphere* args = x y z R
       *cylinder* args = dim c1 c2 R
       *cone* args = dim c1 c2 radlo radhi lo hi
       *plane* args = dim pos side
* zero or more keyword/value pairs may be appended
* keyword = *side* or *units*
       *side* value = *in* or *out*
         *in* = the indenter acts on particles inside the sphere or cylinder or cone

*Examples*

```lammps
fix 1 all indent 10.0 sphere 0.0 0.0 15.0 3.0
fix 1 all indent 10.0 sphere v_x v_y 0.0 v_radius side in
fix 2 flow indent 10.0 cylinder z 0.0 0.0 10.0 units box
```

Insert an indenter within a simulation box.  The indenter repels all
atoms in the group that touch it, so it can be used to push into a
material or as an obstacle in a flow.  Alternatively, it can be used as a
constraining wall around a simulation; see the discussion of the
*side* keyword below.

The *gstyle* keyword selects the geometry of the indenter and it can
either have the value of *sphere*, *cylinder*, *cone*, or *plane*.

*Constraints and cautions from this page*

- **NOTE:** If you want the atom/indenter interaction energy to be included in the total potential energy of the system (the quantity being minimized), you must enable the `fix_modify` *energy* option for this fix.

#### `compute stress/atom` — local stress field

*Source: `doc/src/compute_stress_atom.rst`*

```lammps
compute ID group-ID style temp-ID keyword ...
```

* ID, group-ID are documented in `compute` command
* style = *stress/atom* or *centroid/stress/atom*
* temp-ID = ID of compute that calculates temperature, can be NULL if not needed
* zero or more keywords may be appended
* keyword = *ke* or *pair* or *bond* or *angle* or *dihedral* or *improper* or *kspace* or *fix* or *virial*

*Examples*

```lammps
compute 1 mobile stress/atom NULL
compute 1 mobile stress/atom myRamp
compute 1 all stress/atom NULL pair bond
compute 1 all centroid/stress/atom NULL bond dihedral improper
```

Define a computation that computes per-atom stress tensor for each
atom in a group.  In case of compute *stress/atom*, the tensor for
each atom is symmetric with 6 components and is stored as a 6-element
vector in the following order: $xx$, $yy$, $zz$,
$xy$, $xz$, $yz$.  In case of compute
*centroid/stress/atom*, the tensor for each atom is asymmetric with 9
components and is stored as a 9-element vector in the following order:
$xx$, $yy$, $zz$, $xy$, $xz$,
$yz$, $yx$, $zx$, $zy$.  See the `compute
pressure` command if you want the stress tensor
(pressure) of the entire system.

The stress tensor for atom $I$ is given by the following
formula, where $a$ and $b$ take on values $x$,
$y$, $z$ to generate the components of the tensor:

*Constraints and cautions from this page*

- **NOTE:** The per-atom stress does not include any Lennard-Jones tail corrections to the pressure added by the `pair_modify tail yes` command, since those are contributions to the global system pressure.

---

## 4. Gas-surface scattering and accommodation coefficients

*Why this is here:* The thermal accommodation coefficient is, by the project's own spec, the weakest input in the whole model -- measured values for one gas/solid pair span an order of magnitude. LAMMPS provides the FORWARD model (impose a given accommodation) via fix wall/reflect/stochastic with diffusive, Maxwell and Cercignani-Lampis styles. It does NOT measure alpha: for that, fire gas atoms at a real atomistic surface and compare incident with reflected energy against the wall temperature. No canned example exists for the measurement.

### Official scripts

#### `wall/in.wall.diffusive` — Fully diffusive reflection

```lammps
units		metal
dimension	3
boundary	p f p

atom_style	atomic
neighbor	2.0 bin

# create geometry

lattice		sc 10
region		box block 0 10 0 10 0 10
create_box	1 box
create_atoms	1 box

mass		1 39.95

# LJ potentials

pair_style	lj/cut 8.5
pair_coeff	* * 0.01 3.4 8.5

# initial velocities

velocity	all create 300 482748 
fix		1 all nve

# diffusive reflection
# upper wall velocity = 0.1, lower wall fixed

fix ywalls all wall/reflect/stochastic diffusive 2308 &
     ylo EDGE 300 0 0 0 yhi EDGE 300 0.1 0 0

# run

thermo		1000
run		10000
```

#### `wall/in.wall.specular` — Specular reflection

```lammps
units		metal
dimension	3
boundary	p f p

atom_style	atomic
neighbor	2.0 bin

# create geometry

lattice		sc 10
region		box block 0 10 0 10 0 10
create_box	1 box
create_atoms	1 box

mass		1 39.95

# LJ potentials

pair_style	lj/cut 8.5
pair_coeff	* * 0.01 3.4 8.5

# initial velocities

velocity	all create 300 482748 
fix		1 all nve

# specular reflection

fix ywalls all wall/reflect ylo EDGE yhi EDGE

# run

thermo		1000
run		10000
```

#### `wall/in.wall.maxwell` — Maxwell model, one accommodation coeff

```lammps
units		metal
dimension	3
boundary	p f p

atom_style	atomic
neighbor	2.0 bin

# create geometry

lattice		sc 10
region		box block 0 10 0 10 0 10
create_box	1 box
create_atoms	1 box

mass		1 39.95

# LJ potentials

pair_style	lj/cut 8.5
pair_coeff	* * 0.01 3.4 8.5

# initial velocities

velocity	all create 300 482748 
fix		1 all nve

# Maxwellian reflection
# upper wall velocity = 0.1, lower wall fixed, 
# accommodation coefficient = 0.2 for both walls

fix ywalls all wall/reflect/stochastic maxwell 29839 &
     ylo EDGE 300 0 0 0 0.2 yhi EDGE 300 0 0 0.1 0.2 

# run

thermo		1000
run		10000
```

#### `wall/in.wall.ccl` — Cercignani-Lampis, per-direction accommodation coeffs

```lammps
units		metal
dimension	3
boundary	p f p

atom_style	atomic
neighbor	2.0 bin

# create geometry

lattice		sc 10
region		box block 0 10 0 10 0 10
create_box	1 box
create_atoms	1 box

mass		1 39.95

# LJ potentials

pair_style	lj/cut 8.5
pair_coeff	* * 0.01 3.4 8.5

# initial velocities

velocity	all create 300 482748 
fix		1 all nve

# CCL reflection model
# upper wall velocity = 0.1, lower wall fixed,
# accommodation coefficients = 0.1, 0.2 and 0.3 along directions x,y,z

fix ywalls all wall/reflect/stochastic ccl 345533 &
    ylo EDGE 300 0 0 0 0.1 0.2 0.3 yhi EDGE 300 0.1 0 0 0.1 0.2 0.3

# run

thermo		1000
run		10000
```

### Command reference

#### `fix wall/reflect/stochastic` — accommodation models at a wall

*Source: `doc/src/fix_wall_reflect_stochastic.rst`*

```lammps
fix ID group-ID wall/reflect/stochastic rstyle seed face args ... keyword value ...
```

```lammps
args = pos temp velx vely velz accomx accomy accomz
  pos = EDGE or constant
    EDGE = current lo or hi edge of simulation box
    constant = number like 0.0 or 30.0 (distance units)
  temp = wall temperature (temperature units)
  velx,vely,velz = wall velocity in x,y,z directions (velocity units)
  accomx,accomy,accomz = accommodation coeffs in x,y,z directions (unitless)
    not specified for rstyle = diffusive
    single accom coeff specified for rstyle maxwell
    all 3 coeffs specified for rstyle cll
```

```lammps
*units* value = *lattice* or *box*
  *lattice* = the wall position is defined in lattice units
  *box* = the wall position is defined in simulation box units
```

* ID, group-ID are documented in `fix` command
* wall/reflect/stochastic = style name of this fix command
* rstyle = diffusive or maxwell or ccl
* seed = random seed for stochasticity (positive integer)
* one or more face/args pairs may be appended
* face = *xlo* or *xhi* or *ylo* or *yhi* or *zlo* or *zhi*
* zero or more keyword/value pairs may be appended
* keyword = *units*
       *units* value = *lattice* or *box*
         *lattice* = the wall position is defined in lattice units
         *box* = the wall position is defined in simulation box units

*Examples*

```lammps
fix zwalls all wall/reflect/stochastic diffusive 23424 zlo EDGE 300 0.1 0.1 0 zhi EDGE 200 0.1 0.1 0
fix ywalls all wall/reflect/stochastic maxwell 345533 ylo 5.0 300 0.1 0.0 0.0 0.8 yhi 10.0 300 0.1 0.0 0.0 0.8
fix xwalls all wall/reflect/stochastic cercignanilampis 2308 xlo 0.0 300 0.0 0.1 0.9 0.8 0.7 xhi EDGE 300 0.0 0.1 0 0.9 0.8 0.7 units box
```

Bound the simulation with one or more walls which reflect particles
in the specified group when they attempt to move through them.

Reflection means that if an atom moves outside the wall on a timestep
(e.g. due to the `fix nve` command), then it is put back
inside the wall with a changed velocity.

#### `fix wall/reflect` — simple reflecting wall

*Source: `doc/src/fix_wall_reflect.rst`*

```lammps
fix ID group-ID wall/reflect face arg ... keyword value ...
```

```lammps
arg = EDGE or constant or variable
  EDGE = current lo edge of simulation box
  constant = number like 0.0 or 30.0 (distance units)
  variable = :doc:`equal-style variable <variable>` like v_x or v_wiggle
```

```lammps
*units* value = *lattice* or *box*
  *lattice* = the wall position is defined in lattice units
  *box* = the wall position is defined in simulation box units
```

* ID, group-ID are documented in `fix` command
* wall/reflect = style name of this fix command
* one or more face/arg pairs may be appended
* face = *xlo* or *xhi* or *ylo* or *yhi* or *zlo* or *zhi*
* zero or more keyword/value pairs may be appended
* keyword = *units*
       *units* value = *lattice* or *box*
         *lattice* = the wall position is defined in lattice units
         *box* = the wall position is defined in simulation box units

*Examples*

```lammps
fix xwalls all wall/reflect xlo EDGE xhi EDGE
fix walls all wall/reflect xlo 0.0 ylo 10.0 units box
fix top all wall/reflect zhi v_pressdown
```

Bound the simulation with one or more walls which reflect particles
in the specified group when they attempt to move through them.

Reflection means that if an atom moves outside the wall on a timestep
by a distance delta (e.g. due to `fix nve`), then it is
put back inside the face by the same delta, and the sign of the
corresponding component of its velocity is flipped.

#### `fix wall/lj93 etc` — potential-based walls

*Source: `doc/src/fix_wall.rst`*

```lammps
fix ID group-ID style [tabstyle] [N] face args ... keyword value ...
```

```lammps
args = coord epsilon sigma cutoff
coord = position of wall = EDGE or constant or variable
  EDGE = current lo or hi edge of simulation box
  constant = number like 0.0 or -30.0 (distance units)
  variable = :doc:`equal-style variable <variable>` like v_x or v_wiggle
epsilon = strength factor for wall-particle interaction (energy or energy/distance\^2 units)
  epsilon can be a variable (see below)
sigma = size factor for wall-particle interaction (distance units)
  sigma can be a variable (see below)
cutoff = distance from wall at which wall-particle interactions are cut off (distance units)
```

```lammps
args = coord expression cutoff
coord = position of wall = EDGE or constant or variable
  EDGE = current lo or hi edge of simulation box
  constant = number like 0.0 or -30.0 (distance units)
  variable = :doc:`equal-style variable <variable>` like v_x or v_wiggle
expression = Lepton expression for the potential  (energy units)
cutoff = distance from wall at which wall-particle interactions are cut off (distance units)
```

```lammps
args = coord D_0 alpha r_0 cutoff
coord = position of wall = EDGE or constant or variable
  EDGE = current lo or hi edge of simulation box
  constant = number like 0.0 or -30.0 (distance units)
  variable = :doc:`equal-style variable <variable>` like v_x or v_wiggle
D_0 = depth of the potential (energy units)
  D_0 can be a variable (see below)
alpha = width factor for wall-particle interaction (1/distance units)
  alpha can be a variable (see below)
r_0 = distance of the potential minimum from the face of region (distance units)
  r_0 can be a variable (see below)
cutoff = distance from wall at which wall-particle interactions are cut off (distance units)
```

* ID, group-ID are documented in `fix` command
* style = *wall/lj93* or *wall/lj126* or *wall/lj1043* or *wall/colloid* or *wall/harmonic* or *wall/harmonic/outside* or *wall/lepton* or *wall/morse* or *wall/table*
* tabstyle = *linear* or *spline* = method of table interpolation (only applies to *wall/table*)
* N = use N values in *linear* or *spline* interpolation (only applies to *wall/table*)
* one or more face/arg pairs may be appended
* face = *xlo* or *xhi* or *ylo* or *yhi* or *zlo* or *zhi*
* args for styles *lj93* or *lj126* or *lj1043* or *colloid* or *harmonic*
* args for style *lepton*
* args for style *morse*
* args for style *table*
* zero or more keyword/value pairs may be appended
* keyword = *units* or *fld* or *pbc*

*Examples*

```lammps
fix wallhi all wall/lj93 xlo -1.0 1.0 1.0 2.5 units box
fix wallhi all wall/lj93 xhi EDGE 1.0 1.0 2.5
fix wallhi all wall/harmonic xhi EDGE 100.0 0.0 4.0 units box
fix wallhi all wall/morse xhi EDGE 1.0 1.0 1.0 2.5 units box
fix wallhi all wall/lj126 v_wiggle 23.2 1.0 1.0 2.5
fix zwalls all wall/colloid zlo 0.0 1.0 1.0 0.858 zhi 40.0 1.0 1.0 0.858
fix xwall mobile wall/table spline 200 EDGE -5.0 walltab.dat HARMONIC 4.0
fix xwalls mobile wall/lepton xlo -5.0 "k*(r-rc)^2;k=100.0" 4.0 xhi 5.0 "k*(r-rc)^2;k=100.0" 4.0
```

Bound the simulation domain on one or more of its faces with a flat
wall that interacts with the atoms in the group by generating a force
on the atom in a direction perpendicular to the wall.  The energy of
wall-particle interactions depends on the style.

For style *wall/lj93*, the energy E is given by the 9-3 Lennard-Jones potential:

*Constraints and cautions from this page*

- **NOTE:** For all of the styles (except wall/harmonic/outside), you must ensure that r is always > 0 for all particles in the group, or LAMMPS will generate an error.  This means you cannot start your simulation with particles at the wall position *coord* (r = 0) or with particles on the wrong side of the wall (r < 0).  For the *wall/lj93* and *wall/lj126* styles, the energy of the wall/particle interaction (and hence the force on the particle) blows up as r -> 0.  The *wall/colloid* style is even more restrictive, since the energy blows up as D = r-R -> 0.  This means the finite-size particles of radius R must be a distance larger than R from the wall position *coord*. The *harmonic* style is a softer potential and does not blow up as r -> 0, but you must use a large enough $\epsilon$ that particles always remain on the correct side of the wall (r > 0).
- **NOTE:** If you want the atom/wall interaction energy to be included in the total potential energy of the system (the quantity being minimized), you MUST enable the `fix_modify` *energy* option for this fix.

#### `fix wall/region` — wall from a region surface

*Source: `doc/src/fix_wall_region.rst`*

```lammps
fix ID group-ID wall/region region-ID style args ... cutoff
```

```lammps
epsilon = strength factor for wall-particle interaction (energy or energy/distance\^2 units)
sigma = size factor for wall-particle interaction (distance units)
```

```lammps
D_0 = depth of the potential (energy units)
alpha = width parameter (1/distance units)
r_0 = distance of the potential minimum from wall position (distance units)
```

* ID, group-ID are documented in `fix` command
* wall/region = style name of this fix command
* region-ID = region whose boundary will act as wall
* style = *lj93* or *lj126* or *lj1043* or *colloid* or *harmonic* or *morse*
* args for styles *lj93* or *lj126* or *lj1043* or *colloid* or *harmonic* =
* args for style *morse* =
* cutoff = distance from wall at which wall-particle interaction is cut off (distance units)

*Examples*

```lammps
fix wall all wall/region mySphere lj93 1.0 1.0 2.5
fix wall all wall/region mySphere harmonic 1.0 0.0 2.5
fix wall all wall/region box_top morse 1.0 1.0 1.5 3.0
```

Treat the surface of the geometric region defined by the *region-ID*
as a bounding wall which interacts with nearby particles according to
the specified style.

The distance between a particle and the surface is the distance to the
nearest point on the surface and the force the wall exerts on the
particle is along the direction between that point and the particle,
which is the direction normal to the surface at that point.  Note that
if the region surface is comprised of multiple "faces", then each face
can exert a force on the particle if it is close enough.  E.g. for
`region_style block`, a particle in the interior, near a
corner of the block, could feel wall forces from 1, 2, or 3 faces of
the block.

*Constraints and cautions from this page*

- **NOTE:** As discussed on the `region` command doc page, regions in LAMMPS do not get wrapped across periodic boundaries.  It is up to you to ensure that periodic or non-periodic boundaries are specified appropriately via the `boundary` command when using a region as a wall that bounds particle motion.  This also means that if you embed a region in your simulation box and want it to repulse particles from its surface (using the "side out" option in the `region` command), that its repulsive force will not be felt across a periodic boundary.
- **NOTE:** For primitive regions with sharp corners and/or edges (e.g. a block or cylinder), wall/particle forces are computed accurately for both interior and exterior regions.  For *union* and *intersect* regions, additional sharp corners and edges may be present due to the intersection of the surfaces of 2 or more primitive volumes.  These corners and edges can be of two types: concave or convex.  Concave points/edges are like the corners of a cube as seen by particles in the interior of a cube.  Wall/particle forces around these features are computed correctly.  Convex points/edges are like the corners of a cube as seen by particles exterior to the cube, i.e. the points jut into the volume where particles are present.  LAMMPS does NOT compute the location of these convex points directly, and hence wall/particle forces in the cutoff volume around these points suffer from inaccuracies.  The basic problem is that the outward normal of the surface is not continuous at these points.  This can cause particles to feel no force (they don't "see" the wall) when in one location, then move a distance epsilon, and suddenly feel a large force because they now "see" the wall.  In a worst-case scenario, this can blow particles out of the simulation box.  Thus, as a general rule you should not use the fix wall/gran/region command with *union* or *intersect* regions that have convex points or edges resulting from the union/intersection (convex points/edges in the union/intersection due to a single sub-region are still OK).
- **NOTE:** Similarly, you should not define *union* or *intersert* regions for use with this command that share an overlapping common face that is part of the overall outer boundary (interior boundary is OK), even if the face is smooth.  E.g. two regions of style block in a *union* region, where the two blocks overlap on one or more of their faces. This is because LAMMPS discards points that are part of multiple sub-regions when calculating wall/particle interactions, to avoid double-counting the interaction.  Having two coincident faces could cause the face to become invisible to the particles.  The solution is to make the two faces differ by epsilon in their position.
- **NOTE:** If you want the atom/wall interaction energy to be included in the total potential energy of the system (the quantity being minimized), you MUST enable the `fix_modify` *energy* option for this fix.

---

## 5. Surface structure and roughness

*Why this is here:* Nanoscale roughness and defects at the interface are an optional part of the MD step, and the Phase-2 structure module needs to cut, stack and perturb surfaces. There is no official 'roughness' example; these are the commands that build one.

### Command reference

#### `displace_atoms` — move atoms, including random displacement

*Source: `doc/src/displace_atoms.rst`*

```lammps
displace_atoms group-ID style args keyword value ...
```

```lammps
*move* args = delx dely delz
  delx,dely,delz = distance to displace in each dimension (distance units)
  any of delx,dely,delz can be a variable (see below)
*ramp* args = ddim dlo dhi dim clo chi
  ddim = *x* or *y* or *z*
  dlo,dhi = displacement distance between dlo and dhi (distance units)
  dim = *x* or *y* or *z*
  clo,chi = lower and upper bound of domain to displace (distance units)
*random* args = dx dy dz seed
  dx,dy,dz = random displacement magnitude in each dimension (distance units)
  seed = random # seed (positive integer)
*rotate* args = Px Py Pz Rx Ry Rz theta
  Px,Py,Pz = origin point of axis of rotation (distance units)
  Rx,Ry,Rz = axis of rotation vector
  theta = angle of rotation (degrees)
```

```lammps
keyword = *units*
  *units* value = *box* or *lattice*
```

* group-ID = ID of group of atoms to displace
* style = *move* or *ramp* or *random* or *rotate*
       *move* args = delx dely delz
       *ramp* args = ddim dlo dhi dim clo chi
       *random* args = dx dy dz seed
       *rotate* args = Px Py Pz Rx Ry Rz theta
* zero or more keyword/value pairs may be appended
         *units* value = *box* or *lattice*

*Examples*

```lammps
displace_atoms top move 0 -5 0 units box
displace_atoms flow ramp x 0.0 5.0 y 2.0 20.5
```

Displace a group of atoms.  This can be used to move atoms a large
distance before beginning a simulation or to randomize atoms initially
on a lattice.  For example, in a shear simulation, an initial strain
can be imposed on the system.  Or two groups of atoms can be brought
into closer proximity.

The *move* style displaces the group of atoms by the specified 3d
displacement vector.  Any of the three quantities defining the vector
components can be specified as an equal-style or atom-style
`variable`.  If the value is a variable, it should be
specified as v_name, where name is the variable name.  In this case,
the variable will be evaluated, and its value(s) used for the
displacement(s).  The scale factor implied by the *units* keyword will
also be applied to the variable result.

*Constraints and cautions from this page*

- **NOTE:** Care should be taken not to move atoms on top of other atoms. After the move, atoms are remapped into the periodic simulation box if needed, and any shrink-wrap boundary conditions (see the `boundary` command) are enforced which may change the box size.  Other than this effect, this command does not change the size or shape of the simulation box.  See the `change_box` command if that effect is desired.
- **NOTE:** Atoms can be moved arbitrarily long distances by this command. If the simulation box is non-periodic and shrink-wrapped (see the `boundary` command), this can change its size or shape. This is not a problem, except that the mapping of processors to the simulation box is not changed by this command from its initial 3d configuration; see the `processors` command.  Thus, if the box size/shape changes dramatically, the mapping of processors to the simulation box may not end up as optimal as the initial mapping attempted to be.

#### `delete_atoms` — remove atoms, including random

*Source: `doc/src/delete_atoms.rst`*

```lammps
delete_atoms style args keyword value ...
```

```lammps
*group* args = group-ID
*region* args = region-ID
*overlap* args = cutoff group1-ID group2-ID
  cutoff = delete one atom from pairs of atoms within the cutoff (distance units)
  group1-ID = one atom in pair must be in this group
  group2-ID = other atom in pair must be in this group
*random* args = ranstyle value eflag group-ID region-ID seed
  ranstyle = *fraction* or *count*
    for *fraction*:
      value = fraction (0.0 to 1.0) of eligible atoms to delete
      eflag =  *no* for fast approximate deletion, *yes* for exact deletion
    for *count*:
      value = number of atoms to delete
      eflag = *no* for warning if count > eligible atoms, *yes* for error
  group-ID = group within which to perform deletions
  region-ID = region within which to perform deletions
              or NULL to only impose the group criterion
  seed = random number seed (positive integer)
*variable* args = variable-name
```

```lammps
*compress* value = *no* or *yes*
*condense* value = *no* or *yes*
*bond* value = *no* or *yes*
*mol* value = *no* or *yes*
```

* style = *group* or *region* or *overlap* or *random* or *variable*
       *group* args = group-ID
       *region* args = region-ID
       *overlap* args = cutoff group1-ID group2-ID
       *random* args = ranstyle value eflag group-ID region-ID seed
       *variable* args = variable-name
* zero or more keyword/value pairs may be appended
* keyword = *compress* or *condense* or *bond* or *mol*
       *compress* value = *no* or *yes*
       *condense* value = *no* or *yes*
       *bond* value = *no* or *yes*
       *mol* value = *no* or *yes*

*Examples*

```lammps
delete_atoms group edge
delete_atoms region sphere compress no
delete_atoms region sphere condense yes
delete_atoms overlap 0.3 all all
delete_atoms overlap 0.5 solvent colloid
delete_atoms random fraction 0.1 yes all cube 482793 bond yes
delete_atoms random fraction 0.3 no polymer NULL 482793 bond yes
delete_atoms random count 500 no ions NULL 482793
delete_atoms variable checkers
```

Delete the specified atoms.  This command can be used, for example, to
carve out voids from a block of material or to delete created atoms
that are too close to each other (e.g., at a grain boundary).

For style *group*, all atoms belonging to the group are deleted.

*Constraints and cautions from this page*

- **WARNING:** If you delete atoms from a molecular system, you must be careful not to end up with bonded interactions that are stored by remaining atoms but which include deleted atoms.  This will cause LAMMPS to generate a "missing atoms" error when the bonded interaction is computed.  The *bond yes* and *mol yes* settings are recommended to avoid such inconsistencies.
- **NOTE:** The molecule deletion operation is invoked after all individual atoms have been deleted using the rules described above for each style.  This means additional atoms may be deleted that are not in the group or region, that are not required by the overlap cutoff criterion, or that will create a higher fraction of porosity than was requested.

#### `create_atoms` — fill a region from a lattice

*Source: `doc/src/create_atoms.rst`*

```lammps
create_atoms type style args keyword values ...
```

```lammps
*box* args = none
*region* args = region-ID
  region-ID = particles will only be created if contained in the region
*single* args = x y z
  x,y,z = coordinates of a single particle (distance units)
*mesh* args = STL-file
  STL-file = file with triangle mesh in STL format
*random* args = N seed region-ID
  N = number of particles to create
  seed = random # seed (positive integer)
  region-ID = create atoms within this region, use NULL for entire simulation box
```

```lammps
*mol* values = template-ID seed
  template-ID = ID of molecule template specified in a separate :doc:`molecule <molecule>` command
  seed = random # seed (positive integer)
*basis* values = M itype
  M = which basis atom
  itype = atom type (1-Ntypes or type label) to assign to this basis atom
*ratio* values = frac seed
  frac = fraction of lattice sites (0 to 1) to populate randomly
  seed = random # seed (positive integer)
*subset* values = Nsubset seed
  Nsubset = # of lattice sites to populate randomly
  seed = random # seed (positive integer)
*group* value = group name
*remap* value = *yes* or *no*
*var* value = name = variable name to evaluate for test of atom creation
*set* values = dim name
  dim = *x* or *y* or *z*
  name = name of variable to set with x, y, or z atom position
*radscale* value = factor
  factor = scale factor for setting atom radius
*meshmode* values = mode arg
  mode = *bisect* or *qrand*
  *bisect* arg = radthresh
    radthresh = threshold value for *mesh* to determine when to split triangles (distance units)
  *qrand* arg = density
    density = minimum number density for atoms place on *mesh* triangles (inverse distance squared units)
*rotate* values = theta Rx Ry Rz
  theta = rotation angle for single molecule (degrees)
  Rx,Ry,Rz = rotation vector for single molecule
*overlap* value = Doverlap
  Doverlap = only insert if at least this distance from all existing atoms
*maxtry* value = Ntry
  Ntry = number of attempts to insert a particle before failure
*units* value = *lattice* or *box*
  *lattice* = the geometry is defined in lattice units
  *box* = the geometry is defined in simulation box units
```

* type = atom type (1-Ntypes or type label) of atoms to create (offset for molecule creation)
* style = *box* or *region* or *single* or *mesh* or *random*
       *box* args = none
       *region* args = region-ID
       *single* args = x y z
       *mesh* args = STL-file
       *random* args = N seed region-ID
* zero or more keyword/value pairs may be appended
* keyword = *mol* or *basis* or *ratio* or *subset* or *group* or *remap* or *var* or *set* or *radscale* or *meshmode* or *rotate* or *overlap* or *maxtry* or *units*
       *mol* values = template-ID seed
       *basis* values = M itype
       *ratio* values = frac seed

*Examples*

```lammps
create_atoms 1 box

labelmap atom 1 Pt
create_atoms Pt box

labelmap atom 1 C 2 Si
create_atoms C region regsphere basis 1 Si

create_atoms 2 region regsphere basis 2 3
create_atoms 1 region regsphere basis 2 2 ratio 0.5 74637
create_atoms 3 single 0 0 5 group newatom
create_atoms 1 box var v set x xpos set y ypos
create_atoms 2 random 50 12345 NULL overlap 2.0 maxtry 50
create_atoms 1 mesh open_box.stl meshmode qrand 0.1 units box
create_atoms 1 mesh funnel.stl meshmode bisect 4.0 units box radscale 0.9
```

This command creates atoms (or molecules) within the simulation box,
either on a lattice, or at random points, or on a surface defined by a
triangulated mesh.  Or it creates a single atom (or molecule) at a
specified point.  It is an alternative to reading in atom coordinates
explicitly via a `read_data` or `read_restart` command.

To use this command a simulation box must already exist, which is
typically created via the `create_box` command.
Before using this command, a lattice must typically also be defined
using the `lattice` command, unless you specify the
*single* or *mesh* style with units = box or the *random* style.  To
create atoms on a lattice for general triclinic boxes, see the
discussion below.

*Constraints and cautions from this page*

- **NOTE:** LAMMPS allows specification of general triclinic simulation boxes as a convenience for users who may be converting data from solid-state crystallographic representations or from DFT codes for input to LAMMPS.  However, as explained on the `Howto_triclinic` doc page, internally, LAMMPS only uses restricted triclinic simulation boxes.  This means the box created by the `create_box` command as well as the atoms created by this command with their per-atom information (e.g. coordinates, velocities) are converted (rotated) from general to restricted triclinic form when the two commands are invoked.  The `Howto_triclinic` doc page also discusses other LAMMPS commands which can input/output general triclinic representations of the simulation box and per-atom data.
- **NOTE:** The *region* style also operates as explained above, but the check for particles inside the region is performed *after* the particle coordinates have been converted to the restricted triclinic box. This means the region must also be defined with respect to the restricted triclinic box, not the general triclinic box.
- **NOTE:** The atom placement algorithms in the *mesh* style benefit from meshes where triangles are close to equilateral.  It is therefore recommended to pre-process STL files to optimize the mesh accordingly.  There are multiple open source and commercial software tools available with the capability to generate optimized meshes.
- **NOTE:** In most cases the atoms created in *mesh* style will become an immobile or rigid object that would not be time integrated or moved by `fix move` or `fix rigid`.  For computational efficiency *and* to avoid undesired contributions to pressure and potential energy due to close contacts, it is usually beneficial to exclude computing interactions between the created particles using `neigh_modify exclude`.
- **NOTE:** If you are using the *mol* keyword in combination with the `atom style template` command, they must use the same molecule template-ID.
- **NOTE:** If the `create_box` command is used to create the simulation box, followed by the create_atoms command with its *mol* option for adding molecules, then you typically need to use the optional keywords allowed by the `create_box` command for extra bonds (angles,etc) or extra special neighbors.  This is because by default, the `create_box` command sets up a non-molecular system that does not allow molecules to be added.

#### `fix deposit` — add atoms onto a surface

*Source: `doc/src/fix_deposit.rst`*

```lammps
fix ID group-ID deposit N type M seed keyword values ...
```

```lammps
*region* value = region-ID
  region-ID = ID of region to use as insertion volume
*var* value = name = variable name to evaluate for test of atom creation
*set* values = dim name
  dim = *x* or *y* or *z*
  name = name of variable to set with x, y, or z atom position
*id* value = *max* or *next*
  max = atom ID for new atom(s) is max ID of all current atoms plus one
  next = atom ID for new atom(s) increments by one for every deposition
*global* values = lo hi
  lo,hi = put new atom/molecule a distance lo-hi above all other atoms (distance units)
*local* values = lo hi delta
  lo,hi = put new atom/molecule a distance lo-hi above any nearby atom beneath it (distance units)
  delta = lateral distance within which a neighbor is considered "nearby" (distance units)
*near* value = R
  R = only insert atom/molecule if further than R from existing particles (distance units)
*gaussian* values = xmid ymid zmid sigma
  xmid,ymid,zmid = center of the gaussian distribution (distance units)
  sigma = width of gaussian distribution (distance units)
*attempt* value = Q
  Q = attempt a single insertion up to Q times
*rate* value = V
  V = z velocity (y in 2d) at which insertion volume moves (velocity units)
*vx* values = vxlo vxhi
  vxlo,vxhi = range of x velocities for inserted atom/molecule (velocity units)
*vy* values = vylo vyhi
  vylo,vyhi = range of y velocities for inserted atom/molecule (velocity units)
*vz* values = vzlo vzhi
  vzlo,vzhi = range of z velocities for inserted atom/molecule (velocity units)
*target* values = tx ty tz
  tx,ty,tz = location of target point (distance units)
*mol* value = template-ID
  template-ID = ID of molecule template specified in a separate :doc:`molecule <molecule>` command
*molfrac* values = f1 f2 ... fN
  f1 to fN = relative probability of creating each of N molecules in template-ID
*rigid* value = fix-ID
  fix-ID = ID of :doc:`fix rigid/small <fix_rigid>` command
*shake* value = fix-ID
  fix-ID = ID of :doc:`fix shake <fix_shake>` command
*orient* values = rx ry rz
  rx,ry,rz = vector to randomly rotate an inserted molecule around
*units* value = *lattice* or *box*
  lattice = the geometry is defined in lattice units
  box = the geometry is defined in simulation box units
```

* ID, group-ID are documented in `fix` command
* deposit = style name of this fix command
* N = # of atoms or molecules to insert
* type = atom type (1-Ntypes or type label) to assign to inserted atoms (offset for molecule insertion)
* M = insert a single atom or molecule every M steps
* seed = random # seed (positive integer)
* one or more keyword/value pairs may be appended to args
* keyword = *region* or *var* or *set* or *id* or *global* or *local* or *near* or *gaussian* or *attempt* or *rate* or *vx* or *vy* or *vz* or *target* or *mol* or *molfrac* or *rigid* or *shake* or *orient* or *units*
       *region* value = region-ID
       *var* value = name = variable name to evaluate for test of atom creation
       *set* values = dim name
       *id* value = *max* or *next*

*Examples*

```lammps
fix 3 all deposit 1000 2 100 29494 region myblock local 1.0 1.0 1.0 units box
fix 2 newatoms deposit 10000 1 500 12345 region disk near 2.0 vz -1.0 -0.8
fix 4 sputter deposit 1000 2 500 12235 region sphere vz -1.0 -1.0 target 5.0 5.0 0.0 units lattice
fix 5 insert deposit 200 2 100 777 region disk gaussian 5.0 5.0 9.0 1.0 units box

labelmap atom 1 Au
fix 4 sputter deposit 1000 Au 500 12235 region sphere vz -1.0 -1.0 target 5.0 5.0 0.0 units lattice
```

Insert a single atom or molecule into the simulation domain every M
timesteps until N atoms or molecules have been inserted.  This is
useful for simulating deposition onto a surface.  For the remainder of
this doc page, a single inserted atom or molecule is referred to as a
"particle".

If inserted particles are individual atoms, they are assigned the
specified atom type.  If they are molecules, the type of each atom in
the inserted molecule is specified in the file read by the
`molecule` command, and those values are added to the
specified atom type.  E.g. if the file specifies atom types 1,2,3, and
those are the atom types you want for inserted molecules, then specify
*type* = 0.  If you specify *type* = 2, the in the inserted molecule
will have atom types 3,4,5.

*Constraints and cautions from this page*

- **NOTE:** LAMMPS checks that the specified region is wholly inside the simulation box.  It can do this correctly for orthonormal simulation boxes.  However for `triclinic boxes`, it only tests against the larger orthonormal box that bounds the tilted simulation box.  If the specified region includes volume outside the tilted box, then an insertion will likely fail, leading to a "lost atoms" error.  Thus for triclinic boxes you should ensure the specified region is wholly inside the simulation box.
- **NOTE:** If you wish the new rigid molecules (and other rigid molecules) to be thermostatted correctly via `fix rigid/small/nvt` or `fix rigid/small/npt`, then you need to use the `fix_modify dynamic/dof yes` command for the rigid fix.  This is to inform that fix that the molecule count will vary dynamically.
- **NOTE:** If you are inserting finite size particles or a molecule or rigid body consisting of finite-size particles, then you should typically set R larger than the distance at which any inserted particle may overlap with either a previously inserted particle or an existing particle.  LAMMPS will issue a warning if R is smaller than this value, based on the radii of existing and inserted particles.
- **NOTE:** If you are monitoring the temperature of a system where the atom count is changing due to adding particles, you typically should use the `compute_modify dynamic/dof yes` command for the temperature compute you are using.
- **NOTE:** For this to work correctly, the timestep must **not** be changed after reading the restart with `reset_timestep`. The fix will try to detect it and stop with an error.

#### `lattice` — lattice definition and orientation

*Source: `doc/src/lattice.rst`*

```lammps
lattice style scale keyword values ...
```

```lammps
scale = reduced density rho\* (for LJ units)
scale = lattice constant in distance units (for all other units)
```

```lammps
*origin* values = x y z
  x,y,z = fractions of a unit cell (0 <= x,y,z < 1)
*orient* values = dim i j k
  dim = *x* or *y* or *z*
  i,j,k = integer lattice directions
*spacing* values = dx dy dz
  dx,dy,dz = lattice spacings in the x,y,z box directions
*a1*,\ *a2*,\ *a3* values = x y z
  x,y,z = primitive vector components that define unit cell
*basis* values = x y z
  x,y,z = fractional coords of a basis atom (0 <= x,y,z < 1)
*triclinic/general* values = no values
```

* style = *none* or *sc* or *bcc* or *fcc* or *hcp* or *diamond* or         *sq* or *sq2* or *hex* or *custom*
* scale = scale factor between lattice and simulation box
* zero or more keyword/value pairs may be appended
* keyword = *origin* or *orient* or *spacing* or *a1* or *a2* or *a3* or *basis* or *triclinic/general*
       *origin* values = x y z
       *orient* values = dim i j k
       *spacing* values = dx dy dz
       *a1*,*a2*,*a3* values = x y z
       *basis* values = x y z
       *triclinic/general* values = no values

*Examples*

```lammps
lattice fcc 3.52
lattice hex 0.85
lattice sq 0.8 origin 0.0 0.5 0.0 orient x 1 1 0 orient y -1 1 0
lattice custom 3.52 a1 1.0 0.0 0.0 a2 0.5 1.0 0.0 a3 0.0 0.0 0.5 &
                    basis 0.0 0.0 0.0 basis 0.5 0.5 0.5 triclinic/general
lattice none 2.0
```

Define a lattice for use by other commands.  In LAMMPS, a lattice is
simply a set of points in space, determined by a unit cell with basis
atoms, that is replicated infinitely in all dimensions.  The arguments
of the lattice command can be used to define a wide variety of
crystallographic lattices.

A lattice is used by LAMMPS in two ways.  First, the
`create_atoms` command creates atoms on the lattice
points inside the simulation box.  Note that the
`create_atoms` command allows different atom types
to be assigned to different basis atoms of the lattice.  Second, the
lattice spacing in the x,y,z dimensions implied by the lattice, can be
used by other commands as distance units
(e.g. `create_box`, `region` and
`velocity`), which are often convenient to use when the
underlying problem geometry is atoms on a lattice.

*Constraints and cautions from this page*

- **NOTE:** The preceding paragraph describing lattice directions is only valid for orthogonal cubic unit cells (or square in 2d).  If you are using a *hcp* or *hex* lattice or the more general lattice style *custom* with non-orthogonal a1,a2,a3 vectors, then you should think of the 3 *orient* vectors as creating a 3x3 rotation matrix which is applied to a1,a2,a3 to rotate the original unit cell to a new orientation in the simulation box.
- **NOTE:** LAMMPS allows specification of general triclinic lattices and simulation boxes as a convenience for users who may be converting data from solid-state crystallographic representations or from DFT codes for input to LAMMPS.  However, as explained on the `Howto_triclinic` doc page, internally, LAMMPS only uses restricted triclinic simulation boxes.  This means the box and per-atom information (e.g. coordinates, velocities) defined by the `create_box` and `create_atoms` commands are converted from general to restricted triclinic form when the two commands are invoked.  It also means that any other commands which use lattice spacings from this command (e.g. the region command), will be operating on a restricted triclinic simulation box, even if the *triclinic/general* option was used to define the lattice.  See the next section for details.
- **NOTE:** Though they are called lattice spacings, all the commands that have a "units lattice" option, simply use the 3 values as scale factors on the distance units defined by the `units` command.  Thus if you do not like the lattice spacings computed by LAMMPS (e.g. for a non-orthogonal or rotated unit cell), you can define the 3 values to be whatever you wish, via the *spacing* option.
- **NOTE:** If the *triclinic/general* option is specified, the unit cell defined by *a1*, *a2*, *a3* edge vectors is first converted to a restricted triclinic orientation, which is a rotation operation. The min/max extent of the 8 corner points is then determined, as described in the preceding paragraph, to set the lattice spacings. As explained for the *triclinic/general* option above, this is because any use of the lattice spacings by other commands will be for a restricted triclinic simulation box, not a general triclinic box.
- **NOTE:** For non-orthogonal unit cells and/or when a rotation is applied via the *orient* keyword, then the lattice spacings computed by LAMMPS are typically less intuitive.  In particular, in these cases, there is no guarantee that a particular lattice spacing is an integer multiple of the periodicity of the lattice in that direction.  Thus, if you create an orthogonal periodic simulation box whose size in a dimension is a multiple of the lattice spacing, and then fill it with atoms via the `create_atoms` command, you will NOT necessarily create a periodic system.  I.e. atoms may overlap incorrectly at the faces of the simulation box.

#### `region` — geometric regions, units box vs lattice

*Source: `doc/src/region.rst`*

```lammps
region ID style args keyword arg ...
```

```lammps
*delete* = no args
*block* args = xlo xhi ylo yhi zlo zhi
  xlo,xhi,ylo,yhi,zlo,zhi = bounds of block in all dimensions (distance units)
    xlo,xhi,ylo,yhi,zlo,zhi can be a variable (see below)
*cone* args = dim c1 c2 radlo radhi lo hi
  dim = *x* or *y* or *z* = axis of cone
  c1,c2 = coords of cone axis in other 2 dimensions (distance units)
  radlo,radhi = cone radii at lo and hi end (distance units)
  lo,hi = bounds of cone in dim (distance units)
    c1,c2,radlo,radhi,lo,hi can be a variable (see below)
*cylinder* args = dim c1 c2 radius lo hi
  dim = *x* or *y* or *z* = axis of cylinder
  c1,c2 = coords of cylinder axis in other 2 dimensions (distance units)
  radius = cylinder radius (distance units)
    c1,c2, and radius can be a variable (see below)
  lo,hi = bounds of cylinder in dim (distance units)
*ellipsoid* args = x y z a b c
  x,y,z = center of ellipsoid (distance units)
  a,b,c = half the length of the principal axes of the ellipsoid (distance units)
    x,y,z,a,b and c can be a variable (see below)
*plane* args = px py pz nx ny nz
  px,py,pz = point on the plane (distance units)
  nx,ny,nz = direction normal to plane (distance units)
    px,py,pz,nx,ny,nz can be a variable (see below)
*prism* args = xlo xhi ylo yhi zlo zhi xy xz yz
  xlo,xhi,ylo,yhi,zlo,zhi = bounds of untilted prism (distance units)
  xy = distance to tilt y in x direction (distance units)
  xz = distance to tilt z in x direction (distance units)
  yz = distance to tilt z in y direction (distance units)
    xlo,xhi,ylo,yhi,zlo,zhi,xy,xz,yz can be a variable (see below)
*sphere* args = x y z radius
  x,y,z = center of sphere (distance units)
  radius = radius of sphere (distance units)
    x,y,z, and radius can be a variable (see below)
*union* args = N reg-ID1 reg-ID2 ...
  N = # of regions to follow, must be 2 or greater
  reg-ID1,reg-ID2, ... = IDs of regions to join together
*intersect* args = N reg-ID1 reg-ID2 ...
  N = # of regions to follow, must be 2 or greater
  reg-ID1,reg-ID2, ... = IDs of regions to intersect
```

```lammps
*side* value = *in* or *out*
  *in* = the region is inside the specified geometry
  *out* = the region is outside the specified geometry
*units* value = *lattice* or *box*
  *lattice* = the geometry is defined in lattice units
  *box* = the geometry is defined in simulation box units
*move* args = v_x v_y v_z
  v_x,v_y,v_z = equal-style variables for x,y,z displacement of region over time (distance units)
*rotate* args = v_theta Px Py Pz Rx Ry Rz
  v_theta = equal-style variable for rotation of region over time (in radians)
  Px,Py,Pz = origin for axis of rotation (distance units)
  Rx,Ry,Rz = axis of rotation vector
*open* value = integer from 1-6 corresponding to face index (see below)
```

* ID = user-assigned name for the region
* style = *delete* or *block* or *cone* or *cylinder* or *ellipsoid* or *plane* or *prism* or *sphere* or *union* or *intersect*
       *delete* = no args
       *block* args = xlo xhi ylo yhi zlo zhi
       *cone* args = dim c1 c2 radlo radhi lo hi
       *cylinder* args = dim c1 c2 radius lo hi
       *ellipsoid* args = x y z a b c
       *plane* args = px py pz nx ny nz
       *prism* args = xlo xhi ylo yhi zlo zhi xy xz yz
       *sphere* args = x y z radius
       *union* args = N reg-ID1 reg-ID2 ...
       *intersect* args = N reg-ID1 reg-ID2 ...

*Examples*

```lammps
region 1 block -3.0 5.0 INF 10.0 INF INF
region 2 sphere 0.0 0.0 0.0 5 side out
region void cylinder y 2 3 5 -5.0 EDGE units box
region 1 prism 0 10 0 10 0 10 2 0 0
region outside union 4 side1 side2 side3 side4
region 2 sphere 0.0 0.0 0.0 5 side out move v_left v_up NULL
region openbox block 0 10 0 10 0 10 open 5 open 6 units box
region funnel cone z 10 10 2 5 0 10 open 1 units box
```

This command defines a geometric region of space.  Various other
commands use regions.  For example, the region can be filled with atoms
via the `create_atoms` command.  Or a bounding box
around the region, can be used to define the simulation box via the
`create_box` command.  Or the atoms in the region can
be identified as a group via the `group` command, or
deleted via the `delete_atoms` command.  Or the
surface of the region can be used as a boundary wall via the `fix
wall/region` command.

Commands which use regions typically test whether an atom's position
is contained in the region or not.  For this purpose, coordinates
exactly on the region boundary are considered to be interior to the
region.  This means, for example, for a spherical region, an atom on
the sphere surface would be part of the region if the sphere were
defined with the *side in* keyword, but would not be part of the
region if it were defined using the *side out* keyword.  See more
details on the *side* keyword below.

*Constraints and cautions from this page*

- **NOTE:** Regions in LAMMPS do not get wrapped across periodic boundaries, as specified by the `boundary` command.  For example, a spherical region that is defined so that it overlaps a periodic boundary is not treated as 2 half-spheres, one on either side of the simulation box.
- **NOTE:** Regions in LAMMPS are always 3d geometric objects, regardless of whether the `dimension` of a simulation is 2d or 3d. Thus when using regions in a 2d simulation, you should be careful to define the region so that its intersection with the 2d x-y plane of the simulation has the 2d geometric extent you want.
- **NOTE:** Whenever a region property, such as a coordinate or an upper/lower bound, is defined via an equal-style variable, the variable should not cause any of the region boundaries to move too far within a single timestep. Otherwise, bad dynamics will occur.  "Too far" means a small fraction of the approximate distance of closest approach between two particles, which for the case of Lennard-Jones particles is the distance of the energy minimum while for granular particles it is their diameter. An example is a rapidly varying direction vector in region plane since a small change in the normal to plane will shift the region surface far away from the region point by a large displacement.  Similarly, bad dynamics can also occur for fast changing variables employed in the move/rotate options.
- **NOTE:** The *union* and *intersect* regions operate by invoking methods from their list of sub-regions.  Thus you cannot delete the sub-regions after defining a *union* or *intersection* region.
- **NOTE:** Currently, only *block* and *sphere* style regions are supported by KOKKOS.  The code using the region (such as a fix or compute) must also be supported by KOKKOS or no acceleration will occur.

---

## 6. Amorphous structures and melt-quench

*Why this is here:* The Si/SiO2 path. The spec requires reporting the quench rate against experiment (MD 1e11-1e13 K/s against ~1 K/s) and validating density, RDF and bond angles. examples/vashishta carries a real SiO2 parameterisation and a quartz data file.

### Official scripts

#### `melt/in.melt` — Simple melt

```lammps
# 3d Lennard-Jones melt

units           lj
atom_style      atomic

lattice         fcc 0.8442
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create 3.0 87287 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5

neighbor        0.3 bin
neigh_modify    every 20 delay 0 check no

fix             1 all nve

#dump           id all atom 50 dump.melt

#dump           2 all image 25 image.*.jpg type type &
#               axes yes 0.8 0.02 view 60 -30
#dump_modify    2 pad 3

#dump           3 all movie 25 movie.mpg type type &
#               axes yes 0.8 0.02 view 60 -30
#dump_modify    3 pad 3

thermo          50
run             250
```

#### `vashishta/in.vashishta.sio2` — SiO2 with the Vashishta potential

```lammps
# test Vashishta potential for quartz

units		metal
boundary	p p p

atom_style	atomic

read_data	data.quartz

replicate       4 4 4
velocity	all create 2000.0 277387 mom yes
displace_atoms	all move 0.05 0.9 0.4 units box

pair_style 	vashishta
pair_coeff	* *  SiO.1990.vashishta Si O

neighbor	0.3 bin
neigh_modify	delay 10

fix		1 all nve
thermo		10
timestep	0.001

#dump		1 all cfg 10 *.cfg mass type xs ys zs vx vy vz fx fy fz
#dump_modify	1 element Si O

run		100
```

### Command reference

#### `pair_style vashishta` — SiO2 and InP

*Source: `doc/src/pair_vashishta.rst`*

```lammps
pair_style style args
```

```lammps
*vashishta* args = none
*vashishta/table* args = Ntable cutinner
  Ntable = # of tabulation points
  cutinner = tablulate from cutinner to cutoff
```

* style = *vashishta* or *vashishta/table*
* args = list of arguments for a particular style
     *vashishta* args = none
     *vashishta/table* args = Ntable cutinner

*Examples*

```lammps
pair_style vashishta
pair_coeff * * SiC.vashishta Si C

pair_style vashishta/table 100000 0.2
pair_coeff * * SiC.vashishta Si C
```

The *vashishta* and *vashishta/table* styles compute the combined
2-body and 3-body family of potentials developed in the group of Priya
Vashishta and collaborators.  By combining repulsive, screened
Coulombic, screened charge-dipole, and dispersion interactions with a
bond-angle energy based on the Stillinger-Weber potential, this
potential has been used to describe a variety of inorganic compounds,
including SiO2 Vashishta1990, SiC
Vashishta2007, and InP Branicio2009.

The potential for the energy U of a system of atoms is

#### `fix nvt / fix npt` — Nose-Hoover, including temperature ramps for quenching

*Source: `doc/src/fix_nh.rst`*

```lammps
fix ID group-ID style_name keyword value ...
```

```lammps
keyword = *temp* or *iso* or *aniso* or *tri* or *x* or *y* or *z* or *xy* or *yz* or *xz* or *couple* or *tchain* or *pchain* or *mtk* or *tloop* or *ploop* or *nreset* or *drag* or *ptemp* or *dilate* or *scalexy* or *scaleyz* or *scalexz* or *flip* or *isochoric* or *fixedpoint* or *update*
  *temp* values = Tstart Tstop Tdamp
    Tstart,Tstop = external temperature at start/end of run
    Tdamp = temperature damping parameter (time units)
  *iso* or *aniso* or *tri* values = Pstart Pstop Pdamp
    Pstart,Pstop = scalar external pressure at start/end of run (pressure units)
    Pdamp = pressure damping parameter (time units)
  *x* or *y* or *z* or *xy* or *yz* or *xz* values = Pstart Pstop Pdamp
    Pstart,Pstop = external stress tensor component at start/end of run (pressure units)
    Pdamp = stress damping parameter (time units)
  *couple* = *none* or *xyz* or *xy* or *yz* or *xz*
  *tchain* value = N
    N = length of thermostat chain (1 = single thermostat)
  *pchain* value = N
    N length of thermostat chain on barostat (0 = no thermostat)
  *mtk* value = *yes* or *no* = add in MTK adjustment term or not
  *tloop* value = M
    M = number of sub-cycles to perform on thermostat
  *ploop* value = M
    M = number of sub-cycles to perform on barostat thermostat
  *nreset* value = reset reference cell every this many timesteps
  *drag* value = Df
    Df = drag factor added to barostat/thermostat (0.0 = no drag)
  *ptemp* value = Ttarget
    Ttarget = target temperature for barostat
  *dilate* value = dilate-group-ID
    dilate-group-ID = only dilate atoms in this group due to barostat volume changes
  *scalexy* value = *yes* or *no* = scale xy with ly
  *scaleyz* value = *yes* or *no* = scale yz with lz
  *scalexz* value = *yes* or *no* = scale xz with lz
  *flip* value = *yes* or *no* = allow or disallow box flips when it becomes highly skewed
  *isochoric* = *x* or *y* or *z* or *xy* or *yz* or *xz*
  *fixedpoint* values = x y z
    x,y,z = perform barostat dilation/contraction around this point (distance units)
  *update* value = *dipole* or *dipole/dlm*
    dipole = update dipole orientation (only for sphere variants)
    dipole/dlm = use DLM integrator to update dipole orientation (only for sphere variants)
```

* ID, group-ID are documented in `fix` command
* style_name = *nvt* or *npt* or *nph*
* one or more keyword/value pairs may be appended
       *temp* values = Tstart Tstop Tdamp
       *iso* or *aniso* or *tri* values = Pstart Pstop Pdamp
       *x* or *y* or *z* or *xy* or *yz* or *xz* values = Pstart Pstop Pdamp
       *couple* = *none* or *xyz* or *xy* or *yz* or *xz*
       *tchain* value = N
       *pchain* value = N
       *mtk* value = *yes* or *no* = add in MTK adjustment term or not
       *tloop* value = M
       *ploop* value = M

*Examples*

```lammps
fix 1 all nvt temp 300.0 300.0 100.0
fix 1 water npt temp 300.0 300.0 100.0 iso 0.0 0.0 1000.0
fix 2 jello npt temp 300.0 300.0 100.0 tri 5.0 5.0 1000.0
fix 2 ice nph x 1.0 1.0 0.5 y 2.0 2.0 0.5 z 3.0 3.0 0.5 yz 0.1 0.1 0.5 xz 0.2 0.2 0.5 xy 0.3 0.3 0.5 nreset 1000
```

These commands perform time integration on Nose-Hoover style
non-Hamiltonian equations of motion which are designed to generate
positions and velocities sampled from the canonical (nvt),
isothermal-isobaric (npt), and isenthalpic (nph) ensembles.  This
updates the position and velocity for atoms in the group each timestep.

The thermostatting and barostatting is achieved by adding some dynamic
variables which are coupled to the particle velocities (thermostatting)
and simulation domain dimensions (barostatting).  In addition to basic
thermostatting and barostatting, these fixes can also create a chain of
thermostats coupled to the particle thermostat, and another chain of
thermostats coupled to the barostat variables.  The barostat can be
coupled to the overall box volume, or to individual dimensions,
including the *xy*, *xz* and *yz* tilt dimensions.  The external
pressure of the barostat can be specified as either a scalar pressure
(isobaric ensemble) or as components of a symmetric stress tensor
(constant stress ensemble).  When used correctly, the time-averaged
temperature and stress tensor of the particles will match the target
values specified by Tstart/Tstop and Pstart/Pstop.

*Constraints and cautions from this page*

- **NOTE:** A Nose-Hoover thermostat will not work well for arbitrary values of *Tdamp*.  If *Tdamp* is too small, the temperature can fluctuate wildly; if it is too large, the temperature will take a very long time to equilibrate.  A good choice for many models is a *Tdamp* of around 100 timesteps.  Note that this is NOT the same as 100 time units for most `units` settings.  A simple way to ensure this, is via using an `immediate variable` expression accessing the thermo property 'dt', which is the length of the time step.  Example:
- **NOTE:** A Nose-Hoover barostat will not work well for arbitrary values of *Pdamp*.  If *Pdamp* is too small, the pressure and volume can fluctuate wildly; if it is too large, the pressure will take a very long time to equilibrate.  A good choice for many models is a *Pdamp* of around 1000 timesteps.  However, note that *Pdamp* is specified in time units, and that timesteps are NOT the same as time units for most `units` settings.
- **NOTE:** If large strains are caused by the barostat because the initial configuration is far from pressure equilibrium or equilibrated too fast, the system will see large strains on the other dimensions as well.  It is recommended to perform preliminary NPT equilibration if necessary using standard NPT simulations.
- **NOTE:** Using a barostat coupled to tilt dimensions *xy*, *xz*, *yz* can sometimes result in arbitrarily large values of the tilt dimensions, i.e. a dramatically deformed simulation box.  LAMMPS allows the tilt factors to grow a small amount beyond the normal limit of half the box length (0.6 times the box length), and then performs a box "flip" to an equivalent periodic cell.  See the discussion of the *flip* keyword above, to allow this bound to be exceeded, if desired.
- **NOTE:** Unlike the `fix temp/berendsen` command which performs thermostatting but NO time integration, these fixes perform thermostatting/barostatting AND time integration.  Thus you should not use any other time integration fix, such as `fix nve` on atoms to which this fix is applied.  Likewise, fix nvt and fix npt should not normally be used on atoms that also have their temperature controlled by another fix - e.g. by `fix langevin` or `fix temp/rescale` commands.
- **NOTE:** This implementation has been shown to conserve linear momentum up to machine precision under NVT dynamics.  Under NPT dynamics, for a system with zero initial total linear momentum, the total momentum fluctuates close to zero.  It may occasionally undergo brief excursions to non-negligible values, before returning close to zero. Over long simulations, this has the effect of causing the center-of-mass to undergo a slow random walk.  This can be mitigated by resetting the momentum at infrequent intervals using the `fix momentum` command.

#### `compute rdf` — radial distribution function

*Source: `doc/src/compute_rdf.rst`*

```lammps
compute ID group-ID rdf Nbin itype1 jtype1 itype2 jtype2 ... keyword/value ...
```

```lammps
*cutoff* value = Rcut
  Rcut = cutoff distance for RDF computation (distance units)
```

* ID, group-ID are documented in `compute` command
* rdf = style name of this compute command
* Nbin = number of RDF bins
* itypeN = central atom type for Nth RDF histogram (integer, type label, or asterisk form)
* jtypeN = distribution atom type for Nth RDF histogram (integer, type label, or asterisk form)
* zero or more keyword/value pairs may be appended
* keyword = *cutoff*
       *cutoff* value = Rcut

*Examples*

```lammps
compute 1 all rdf 100
compute 1 all rdf 100 1 1
compute 1 all rdf 100 * 3 cutoff 5.0
compute 1 fluid rdf 500 1 1 1 2 2 1 2 2
compute 1 fluid rdf 500 1*3 2 5 *10 cutoff 3.5
```

Define a computation that calculates the radial distribution function
(RDF), also called $g(r)$, and the coordination number for a group of
particles.  Both are calculated in histogram form by binning pairwise
distances into *Nbin* bins from 0.0 to the maximum force cutoff
defined by the `pair_style` command or the cutoff
distance *Rcut* specified via the *cutoff* keyword.  The bins are of
uniform size in radial distance.  Thus a single bin encompasses a thin
shell of distances in 3d and a thin ring of distances in 2d.

If you have a bonded system, then the settings of
   `special_bonds` command can remove pairwise
   interactions between atoms in the same bond, angle, or dihedral.  This
   is the default setting for the `special_bonds`
   command, and means those pairwise interactions do not appear in the
   neighbor list.  Because this fix uses a neighbor list, it also means
   those pairs will not be included in the RDF. This does not apply when
   using long-range coulomb interactions (*coul/long*, *coul/msm*,
   *coul/wolf* or similar.  One way to get around this would be to set
   special_bond scaling factors to very tiny numbers that are not exactly
   zero (e.g., $1.0 \times 10^{-50}$).  Another workaround is to write a
   dump file, and use the `rerun` command to compute the RDF for
   snapshots in the dump file.  The rerun script can use a
   `special_bonds` command that includes all pairs in
   the neighbor list.

*Constraints and cautions from this page*

- **NOTE:** If you have a bonded system, then the settings of `special_bonds` command can remove pairwise interactions between atoms in the same bond, angle, or dihedral.  This is the default setting for the `special_bonds` command, and means those pairwise interactions do not appear in the neighbor list.  Because this fix uses a neighbor list, it also means those pairs will not be included in the RDF. This does not apply when using long-range coulomb interactions (*coul/long*, *coul/msm*, *coul/wolf* or similar.  One way to get around this would be to set special_bond scaling factors to very tiny numbers that are not exactly zero (e.g., $1.0 \times 10^{-50}$).  Another workaround is to write a dump file, and use the `rerun` command to compute the RDF for snapshots in the dump file.  The rerun script can use a `special_bonds` command that includes all pairs in the neighbor list.
- **NOTE:** Normally, you should only use the *cutoff* keyword if no pair style is defined (e.g., the `rerun` command is being used to post-process a dump file of snapshots) or if you really want the RDF for distances beyond the pair_style force cutoff and cannot easily post-process a dump file to calculate it.  This is because using the *cutoff* keyword incurs extra computation and possibly communication, which may slow down your simulation.  If you specify *Rcut* $\le$ force cutoff, you will force an additional neighbor list to be built at every timestep this command is invoked (or every reneighboring timestep, whichever is less frequent), which is inefficient.  LAMMPS will warn you if this is the case.  If you specify a *Rcut* > force cutoff, you must ensure ghost atom information out to *Rcut* + *skin* is communicated, via the `comm_modify cutoff` command, else the RDF computation cannot be performed, and LAMMPS will give an error message.  The *skin* value is what is specified with the `neighbor` command.  In this case, you are forcing a large neighbor list to be built just for the RDF computation, and extra communication to be performed every timestep.
- **NOTE:** compute rdf can handle dynamic groups and systems where atoms are added or removed, but this causes that certain normalization parameters need to be re-computed in every step and include collective communication operations. This will reduce performance and limit parallel efficiency and scaling. For systems, where only the type of atoms changes (e.g., when using `fix atom/swap`), you need to explicitly request the dynamic normalization updates via `compute_modify dynamic/dof yes`

---

## 7. Electron-phonon coupling (two-temperature model)

*Why this is here:* The spec flags Si/Al as a metal-semiconductor interface where electron-phonon coupling carries part of the heat, and states that nothing in the workflow models it. LAMMPS does: fix ttm and fix ttm/mod implement a two-temperature model, and examples/ttm ships an Si parameterisation. This section exists so that caveat can become a plan.

### Official scripts

#### `ttm/in.ttm` — Two-temperature model

```lammps
units           metal
atom_style      atomic
boundary        p p p

variable        latc  equal 2.87
lattice         bcc ${latc}
variable        xmax equal   10.0
variable        xmin equal  -10.0
variable        ymax equal   10.0
variable        ymin equal  -10.0
variable        zmax equal   10.0
variable        zmin equal  -10.0

region          sim_box block ${xmin} ${xmax} ${ymin} ${ymax} ${zmin} ${zmax} &
                units lattice
create_box      1 sim_box
region          atom_box block ${xmin} ${xmax} ${ymin} ${ymax} ${zmin} ${zmax} &
                units lattice
create_atoms    1 region atom_box

mass            1 55.845

pair_style      eam/fs
pair_coeff      * * Fe_mm.eam.fs Fe

neighbor        2.0 bin
neigh_modify    every 5 delay 0 check yes

fix             1 all nve

fix             twotemp all ttm 342785 1.2470e-5 0.087614 &
                0.005365 29.5917 47.5679 58.4613 10 10 10 set 1800.0

compute         pe all pe/atom
compute         ke all ke/atom

timestep        0.0001
thermo          100

thermo_style    custom step temp etotal f_twotemp[1] f_twotemp[2]
                thermo_modify format float "%20.16g"

run             1000
```

#### `ttm/in.ttm.thermal` — TTM with a thermal gradient

```lammps
units           metal
atom_style      atomic
boundary        p p p

variable        latc  equal 2.87
lattice         bcc ${latc}
variable        xmax equal   10.0
variable        xmin equal  -10.0
variable        ymax equal   10.0
variable        ymin equal  -10.0
variable        zmax equal   10.0
variable        zmin equal  -10.0

region          sim_box block ${xmin} ${xmax} ${ymin} ${ymax} ${zmin} ${zmax} &
                units lattice
create_box      1 sim_box
region          atom_box block ${xmin} ${xmax} ${ymin} ${ymax} ${zmin} ${zmax} &
                units lattice
create_atoms    1 region atom_box

mass            1 55.845

pair_style      eam/fs
pair_coeff      * * Fe_mm.eam.fs Fe

neighbor        2.0 bin
neigh_modify    every 5 delay 0 check yes

fix             1 all nve

fix             twotemp all ttm/thermal 11111 props.in 10 10 10 set 1800.0

compute         pe all pe/atom
compute         ke all ke/atom

timestep        0.0001
thermo          100

thermo_style    custom step temp etotal f_twotemp[1] f_twotemp[2]
                thermo_modify format float "%20.16g"

run             1000
```

#### `ttm/in.ttm.mod` — fix ttm/mod variant

```lammps
units           metal
atom_style      atomic
boundary        p p p

lattice diamond 5.4309
region box block 0 10 0 10 0 10
create_box 1 box
mass 1 28.0855
create_atoms 1 box basis 1 1 basis 2 1 basis 3 1 basis 4 1 basis 5 1 basis 6 1 basis 7 1 basis 8 1

pair_style      sw
pair_coeff      * * Si.sw Si

neighbor        2.0 bin
neigh_modify    every 5 delay 0 check yes

fix             1 all nve
fix             twotemp all ttm/mod 1354684 Si.ttm_mod 10 10 10 set 1000.0 # outfile 100 T_out.txt

compute         pe all pe/atom
compute         ke all ke/atom

timestep        0.0001
thermo          100

thermo_style    custom step temp etotal f_twotemp[1] f_twotemp[2]
                thermo_modify format float "%20.16g"

run             1000
```

### Command reference

#### `fix ttm` — two-temperature electron-phonon model

*Source: `doc/src/fix_ttm.rst`*

```lammps
fix ID group-ID ttm seed C_e rho_e kappa_e gamma_p gamma_s v_0 Nx Ny Nz keyword value ...
fix ID group-ID ttm/mod seed init_file Nx Ny Nz keyword value ...
fix ID group-ID ttm/thermal seed properties_file Nx Ny Nz keyword value ...
```

```lammps
C_e  = electronic specific heat (energy/(electron\*temperature) units)
rho_e = electronic density (electrons/volume units)
kappa_e = electronic thermal conductivity (energy/(time\*distance\*temperature) units)
gamma_p = friction coefficient due to electron-ion interactions (mass/time units)
gamma_s = friction coefficient due to electronic stopping (mass/time units)
v_0 = electronic stopping critical velocity (velocity units)
Nx = number of thermal solve grid points in the x-direction (positive integer)
Ny = number of thermal solve grid points in the y-direction (positive integer)
Nz = number of thermal solve grid points in the z-direction (positive integer)
```

```lammps
init_file = file with the parameters to TTM
Nx = number of thermal solve grid points in the x-direction (positive integer)
Ny = number of thermal solve grid points in the y-direction (positive integer)
Nz = number of thermal solve grid points in the z-direction (positive integer)
```

```lammps
properties_file = file with grid based TTM properties
Nx = number of thermal solve grid points in the x-direction (positive integer)
Ny = number of thermal solve grid points in the y-direction (positive integer)
Nz = number of thermal solve grid points in the z-direction (positive integer)
```

* ID, group-ID are documented in `fix` command
* style = *ttm* or *ttm/grid* or *ttm/mod* or *ttm/thermal* or *ttm/cascade*
* seed = random number seed to use for white noise (positive integer)
* remaining arguments for fix *ttm*, fix *ttm/grid*, or fix *ttm/cascade*
* remaining arguments for fix *ttm/mod*:
* remaining arguments for fix *ttm/thermal*:
* zero or more keyword/value(s) pairs may be appended
* keyword = *set* or *infile* or *outfile*
       *set* value = Tinit
       *infile* value = file.in with grid values for electronic temperatures
       *outfile* values = Nout file.out
* fix *ttm/thermal* supports an additional keyword: *source*

*Examples*

```lammps
fix 2 all ttm 699489 1.0 1.0 10 0.1 0.0 2.0 1 12 1 infile initial outfile 1000 T.out
fix 3 all ttm/grid 123456 1.0 1.0 1.0 1.0 1.0 5.0 5 5 5 infile Te.in
fix 4 all ttm/mod 34277 parameters.txt 5 5 5 infile T_init outfile 10 T_out
fix 5 all ttm/thermal 11111 properties.in 10 10 10 source 0.1 infile temps.in outfile 10 temps.out
fix 6 all ttm/cascade 123456 1.0 1.0 1.0 1.0 1.0 5.0 5 5 5 infile Te.in cutoff cetab cetab.dat ketab ketab.dat
```

Use a two-temperature model (TTM) to represent heat transfer through and
between electronic and atomic subsystems.  LAMMPS models the atomic
subsystem as usual with a molecular dynamics model and the classical
force field specified by the user.  The electronic subsystem is modeled
as a continuum, or a background "gas", on a regular grid which overlays
the simulation domain.  Energy can be transferred spatially within the
grid representing the electrons.  Energy can also be transferred between
the electronic and atomic subsystems.  The algorithm underlying this fix
was derived by D. M.  Duffy and A. M. Rutherford and is discussed in two
J Physics: Condensed Matter papers: (Duffy) and
(Rutherford).  They used this algorithm in cascade
simulations where a primary knock-on atom (PKA) was initialized with a
high velocity to simulate a radiation event.

The description in this subsection applies to all 5 fix styles:
*ttm*, *ttm/grid*, *ttm/mod*, *ttm/thermal*, and *ttm/cascade*.

*Constraints and cautions from this page*

- **NOTE:** These fixes do not perform time integration of the atoms in the fix group, they only rescale their velocities.  Thus a time integration fix such as `fix nve` should be used in conjunction with these fixes.  These fixes should not normally be used on atoms that have their temperature controlled by another thermostatting fix, e.g. `fix nvt` or `fix langevin`.
- **NOTE:** These fixes require use of an orthogonal 3d simulation box with periodic boundary conditions in all dimensions.  They also require that the size and shape of the simulation box do not vary dynamically, e.g. due to use of the `fix npt` command. Likewise, the size/shape of processor subdomains cannot vary due to dynamic load-balancing via use of the `fix balance` command.  It is possible however to load balance before the simulation starts using the `balance` command, so that each processor has a different size subdomain.
- **NOTE:** The electronic temperature at each grid point must be a non-zero positive value, both initially, and as the temperature evolves over time.  Thus you must use either the *set* or *infile* keyword or be restarting a simulation that used this fix previously.
- **NOTE:** The fix *ttm/grid* and fix *ttm/cascade* commands do not support the *outfile* keyword. Instead you can use the `dump grid` command to output the electronic temperature on the distributed grid to a dump file or the `restart` command which creates a file specific to the fix which the `read restart` command reads.  The file has the same format as the file the *infile* option reads.
- **NOTE:** The *lsurface* and *rsurface* parameters always define the range of electronic grid cells in the x-direction over which the electron-ion (Langevin) coupling and any external (laser) source term are applied, regardless of the *surface_movement* setting.  Only atoms in grid cells whose x grid index *ix* satisfies *lsurface* <= *ix* < *rsurface* are coupled to the electronic subsystem; atoms in cells outside this range exchange no energy with the electrons.  Therefore, to model a bulk system with full periodicity and no vacuum (i.e. to reproduce the behavior of fix *ttm*), you must set *lsurface* = 0 and *rsurface* = *Nx* so that the coupling acts on the entire simulation box.  Setting *rsurface* to a smaller value couples only a sub-slab of the grid cells and will make energy transfer between the two subsystems appear much too slow.  Fix *ttm/mod* prints a warning when this coupling region does not span the whole box (with *surface_movement* disabled).
- **NOTE:** In the *cutoff* scheme, because particles with velocities above the critical velocity $v_0$ are decoupled from the Langevin force, their kinetic energy should be excluded when measuring the local atomic temperature of each grid cell.

---

## 8. Gas transport properties

*Why this is here:* The gap model derives the molecular mean free path from viscosity rather than a tabulated collision diameter, because viscosity data is better established. These are the LAMMPS routes to viscosity and diffusivity if the gas properties ever need computing rather than looking up.

### Command reference

#### `fix viscosity` — Muller-Plathe shear viscosity

*Source: `doc/src/fix_viscosity.rst`*

```lammps
fix ID group-ID viscosity N vdim pdim Nbin keyword value ...
```

```lammps
*swap* value = Nswap = number of swaps to perform every N steps
*vtarget* value = V or INF = target velocity of swap partners (velocity units)
```

* ID, group-ID are documented in `fix` command
* viscosity = style name of this fix command
* N = perform momentum exchange every N steps
* vdim = *x* or *y* or *z* = which momentum component to exchange
* pdim = *x* or *y* or *z* = direction of momentum transfer
* Nbin = # of layers in pdim direction (must be even number)
* zero or more keyword/value pairs may be appended
* keyword = *swap* or *vtarget*
       *swap* value = Nswap = number of swaps to perform every N steps
       *vtarget* value = V or INF = target velocity of swap partners (velocity units)

*Examples*

```lammps
fix 1 all viscosity 100 x z 20
fix 1 all viscosity 50 x z 20 swap 2 vtarget 1.5
```

Use the Muller-Plathe algorithm described in this paper to exchange momenta between two particles in
different regions of the simulation box every N steps.  This induces a
shear velocity profile in the system.  As described below this enables
a viscosity of the fluid to be calculated.  This algorithm is
sometimes called a reverse non-equilibrium MD (reverse NEMD) approach
to computing viscosity.  This is because the usual NEMD approach is to
impose a shear velocity profile on the system and measure the response
via an off-diagonal component of the stress tensor, which is
proportional to the momentum flux.  In the Muller-Plathe method, the
momentum flux is imposed, and the shear velocity profile is the
system's response.

The simulation box is divided into *Nbin* layers in the *pdim*
direction, where the layer 1 is at the low end of that dimension and
the layer *Nbin* is at the high end.  Every N steps, Nswap pairs of
atoms are chosen in the following manner.  Only atoms in the fix group
are considered.  Nswap atoms in layer 1 with positive velocity
components in the *vdim* direction closest to the target value *V* are
selected.  Similarly, Nswap atoms in the "middle" layer (see below) with
negative velocity components in the *vdim* direction closest to the
negative of the target value *V* are selected.  The two sets of Nswap
atoms are paired up and their *vdim* momenta components are swapped
within each pair.  This resets their velocities, typically in opposite
directions.  Over time, this induces a shear velocity profile in the
system which can be measured using commands such as the following,
which writes the profile to the file tmp.profile:

*Constraints and cautions from this page*

- **NOTE:** If your system is periodic in the direction of the momentum flux, then the flux is going in 2 directions.  This means the effective momentum flux in one direction is reduced by a factor of 2. You will see this in the equations for viscosity in the Muller-Plathe paper.  LAMMPS is simply tallying momentum which does not account for whether or not your system is periodic; you must use the value appropriately to yield a viscosity for your system.
- **NOTE:** After equilibration, if the velocity profile you observe is not linear, then you are likely swapping momentum too frequently and are not in a regime of linear response.  In this case you cannot accurately infer a viscosity and should try increasing the Nevery parameter.

#### `compute viscosity/cos` — periodic-perturbation viscosity

*Source: `doc/src/compute_viscosity_cos.rst`*

```lammps
compute ID group-ID viscosity/cos
```

* ID, group-ID are documented in `compute` command
* viscosity/cos = style name of this compute command

*Examples*

```lammps
units    real
compute  cos all viscosity/cos
variable V equal c_cos[7]
variable A equal 0.02E-5  # A/fs^2
variable density equal density
variable lz equal lz
variable reciprocalViscosity equal v_V/${A}/v_density*39.4784/v_lz/v_lz*100  # 1/(Pa*s)
```

Define a computation that calculates the velocity amplitude of a group of atoms
with an cosine-shaped velocity profile and the temperature of them
after subtracting out the velocity profile before computing the kinetic energy.
A compute of this style can be used by any command that computes a temperature
(e.g., `thermo_modify`, `fix npt`).

This command together with `fix_accelerate/cos`
enables viscosity calculation with periodic perturbation method,
as described by Hess.
An acceleration along the $x$-direction is applied to the simulation
system by using `fix_accelerate/cos` command.
The acceleration is a periodic function along the $z$-direction:

#### `compute msd` — mean-squared displacement

*Source: `doc/src/compute_msd.rst`*

```lammps
compute ID group-ID msd keyword values ...
```

```lammps
*com* value = *yes* or *no*
*average* value = *yes* or *no*
```

* ID, group-ID are documented in `compute` command
* msd = style name of this compute command
* zero or more keyword/value pairs may be appended
* keyword = *com* or *average*
       *com* value = *yes* or *no*
       *average* value = *yes* or *no*

*Examples*

```lammps
compute 1 all msd
compute 1 upper msd com yes average yes
```

Define a computation that calculates the mean-squared displacement
(MSD) of the group of atoms, including all effects due to atoms
passing through periodic boundaries.  For computation of the non-Gaussian
parameter of mean-squared displacement, see the
`compute msd/nongauss` command.

A vector of four quantities is calculated by this compute.  The first three
elements of the vector are the squared *dx*, *dy*, and *dz* displacements,
summed and averaged over atoms in the group.  The fourth element is the total
squared displacement (i.e., $dx^2 + dy^2 + dz^2$), summed and
averaged over atoms in the group.

*Constraints and cautions from this page*

- **NOTE:** Initial coordinates are stored in "unwrapped" form, by using the image flags associated with each atom.  See the `dump custom` command for a discussion of "unwrapped" coordinates.  See the Atoms section of the `read_data` command for a discussion of image flags and how they are set for each atom.  You can reset the image flags (e.g. to 0) before invoking this compute by using the `set image` command.
- **NOTE:** If you want the quantities calculated by this compute to be continuous when running from a `restart file`, then you should use the same ID for this compute, as in the original run. This is so that the fix this compute creates to store per-atom quantities will also have the same ID, and thus be initialized correctly with atom reference positions from the restart file.  When *average* is set to yes, then the atom reference positions are restored correctly, but not the number of samples used obtain them. As a result, the reference positions from the restart file are combined with subsequent positions as if they were from a single sample, instead of many, which will change the values of msd somewhat.

#### `fix nvt/sllod` — SLLOD shear for viscosity

*Source: `doc/src/fix_nvt_sllod.rst`*

```lammps
fix ID group-ID nvt/sllod keyword value ...
```

```lammps
keyword = *psllod* or *peculiar* or *kick* or *integrator*
  *psllod* value = *no* or *yes* = use SLLOD or p-SLLOD variant, respectively
  *peculiar* value = *no* or *yes* = store velocity in the lab frame or peculiar frame, respectively
  *kick* value = *no* or *yes* = whether to superimpose streaming velocity
  *integrator* value = *reversible* or *legacy*
    reversible = use reversible integration scheme
    legacy = use old integration scheme (see below for details)
```

* ID, group-ID are documented in `fix` command
* nvt/sllod = style name of this fix command
* zero or more keyword/value pairs may be appended
      *psllod* value = *no* or *yes* = use SLLOD or p-SLLOD variant, respectively
      *peculiar* value = *no* or *yes* = store velocity in the lab frame or peculiar frame, respectively
      *kick* value = *no* or *yes* = whether to superimpose streaming velocity
      *integrator* value = *reversible* or *legacy*
* additional thermostat related keyword/value pairs from the `fix nvt`

*Examples*

```lammps
fix 1 all nvt/sllod temp 300.0 300.0 100.0
fix 1 all nvt/sllod temp 300.0 300.0 100.0 drag 0.2
```

Perform constant NVT integration to update positions and velocities
each timestep for atoms in the group using a Nose/Hoover temperature
thermostat.  V is volume; T is temperature.  This creates a system
trajectory consistent with the canonical ensemble.

This thermostat is used for a simulation box that is changing size
and/or shape, for example in a non-equilibrium MD (NEMD) simulation.
The size/shape change is induced by use of the `fix deform` command, so each point in the simulation box can be
thought of as having a "streaming" velocity.  This position-dependent
streaming velocity is subtracted from each atom's actual velocity to
yield a thermal velocity which is used for temperature computation and
thermostatting.  For example, if the box is being sheared in x,
relative to y, then points at the bottom of the box (low y) have a
small x velocity, while points at the top of the box (hi y) have a
large x velocity.  These velocities do not contribute to the thermal
"temperature" of the atom.

*Constraints and cautions from this page*

- **NOTE:** `Fix deform` has an option for remapping either atom coordinates or velocities to the changing simulation box.  To use fix nvt/sllod, fix deform should NOT remap atom positions, because fix nvt/sllod adjusts the atom positions and velocities to create a velocity profile that matches the changing box size/shape.  For *peculiar* = *no*, fix deform SHOULD remap atom velocities when atoms cross periodic boundaries since that is consistent with maintaining the velocity profile created by fix nvt/sllod.  For *peculiar* = *yes*, fix deform SHOULD NOT remap velocities.  LAMMPS will give an error if this setting is not consistent.
- **NOTE:** A recent (2017) book by (Daivis and Todd) discusses use of the SLLOD method and non-equilibrium MD (NEMD) thermostatting generally, for both simple and complex fluids, e.g. molecular systems.  The latter can be tricky to do correctly.
- **NOTE:** Some `fix deform` styles like *final*, *scale*, *vel* and *delta* set the deformation rate to zero when "run 0" is called, in which case no kick will be applied for those components of the flow. If a kick is desired in such cases, it is recommended to initialize fix nvt/sllod with "kick no", and then set "kick yes" using `fix_modify` immediately before the next `run` command with non-zero length. Similarly, the *volume*, *wiggle* and *variable* styles do not set a deformation rate during run initialization and are therefore not compatible with the "kick yes" option.

---

## Generation coverage

- example scripts: **20 found**, 0 missing
- command pages: **38 found**, 0 missing
