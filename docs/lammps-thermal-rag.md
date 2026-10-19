# LAMMPS thermal-analysis reference (RAG corpus)

> **Generated — do not edit.** Rebuild with
> `python3 scripts/build_lammps_thermal_rag.py --lammps-src <clone>`.
>
> Source: official LAMMPS source tree, version **30 Sep 2026**.
> `docs.lammps.org` is built from these same `.rst` files, so this is
> the manual's content pinned to the engine actually in use. Scripts
> are emitted verbatim; prose is quoted with its source file named.

**Scope:** the thermal-transport path only — the five official `examples/KAPPA` scripts, the Howto that frames them, the kappa extraction recipes, and reference pages for exactly the commands those scripts use. Nothing else from the manual.

**Retrieval note.** Each command below carries its Syntax block, its official examples, and every `note`/`warning` admonition from its page. The admonitions are the highest-value part: they are where the manual records the constraints that otherwise surface as a failed run.

---

## Overview: calculating thermal conductivity

*Source: `doc/src/Howto_kappa.rst`*

The thermal conductivity $\kappa$ of a material can be measured in at
least 4 ways using various options in LAMMPS.  See the ``examples/KAPPA``
directory for scripts that implement the 4 methods discussed here for
a simple Lennard-Jones fluid model.  Also, see the `Howto viscosity` page for an analogous discussion
for viscosity.

The thermal conductivity tensor $\mathbf{\kappa}$ is a measure of the propensity
of a material to transmit heat energy in a diffusive manner as given
by Fourier's law

.. math::

   J = -\kappa \cdot \text{grad}(T)

where $J$ is the heat flux in units of energy per area per time and
$\text{grad}(T)$ is the spatial gradient of temperature.  The thermal
conductivity thus has units of energy per distance per time per degree
K and is often approximated as an isotropic quantity, i.e. as a
scalar.

The first method is to setup two thermostatted regions at opposite
ends of a simulation box, or one in the middle and one at the end of a
periodic box.  By holding the two regions at different temperatures
with a `thermostatting fix`, the energy added to
the hot region should equal the energy subtracted from the cold region
and be proportional to the heat flux moving between the regions.  See
the papers by Ikeshoji and Hafskjold and
Wirnsberger et al for details of this idea.  Note
that thermostatting fixes such as `fix nvt`, `fix langevin`, and `fix temp/rescale` store the cumulative energy they
add/subtract.

Alternatively, as a second method, the `fix heat` or
`fix ehex` commands can be used in place of thermostats
on each of two regions to add/subtract specified amounts of energy to
both regions.  In both cases, the resulting temperatures of the two
regions can be monitored with the "compute temp/region" command and
the temperature profile of the intermediate region can be monitored
with the `fix ave/chunk` and `compute ke/atom` commands.

The third method is to perform a reverse non-equilibrium MD simulation
using the `fix thermal/conductivity`
command which implements the rNEMD algorithm of Muller-Plathe.
Kinetic energy is swapped between atoms in two different layers of the
simulation box.  This induces a temperature gradient between the two
layers which can be monitored with the `fix ave/chunk` and `compute ke/atom` commands.  The fix tallies the
cumulative energy transfer that it performs.  See the `fix thermal/conductivity` command for
details.

The fourth method is based on the Green-Kubo (GK) formula which
relates the ensemble average of the auto-correlation of the heat flux
to $\kappa$.  The heat flux can be calculated from the fluctuations of
per-atom potential and kinetic energies and per-atom stress tensor in
a steady-state equilibrated simulation.  This is in contrast to the
two preceding non-equilibrium methods, where energy flows continuously
between hot and cold regions of the simulation box.

The `compute heat/flux` command can calculate
the needed heat flux and describes how to implement the Green_Kubo
formalism using additional LAMMPS commands, such as the `fix ave/correlate` command to calculate the needed
auto-correlation.  See the page for the `compute heat/flux` command for an example input script
that calculates the thermal conductivity of solid Ar via the GK
formalism.

---

## Extracting kappa from each method (official recipes)

*Source: `examples/KAPPA/README`, verbatim*

```text
This directory has 5 scripts that compute the thermal conductivity
(kappa) of a Lennard-Jones fluid using 5 different methods.  See the
discussion in the Howto document on calculating thermal conductivity
of the LAMMPS manual for an overview of the methods and pointers to
doc pages for the commands which implement them.  Citations for the
various methods can also be found in the manual.

These scripts are provided for illustration purposes.  No guarantee is
made that the systems are fully equilibrated or that the runs are long
enough to generate good statistics and highly accurate results.

These scripts could easily be adapted to work with solids as well.

-------------

These are the 5 methods for computing thermal conductivity.  The first
4 are non-equilibrium methods; the last is an equilibrium method.

in.langevin = thermostat 2 regions at different temperatures via fix langevin
in.heat = add/subtract energy to 2 regions via fix heat
in.ehex = add/subtract energy to 2 regions via fix ehex
in.mp = use fix thermal/conductivity and the Muller-Plathe method
in.heatflux = use compute heat/flux and the Green-Kubo method

The NEMD systems have 8000 atoms with a box length 2x larger in z, the
non-equilibrium direction.  The G-K system has 4000 atoms and a cubic
box; it also needs to be run longer to generate good statistics.

The scripts were all run on 8 processors.  They all run in a minute or
so and produce the accompanying log files and profile files (for
temperature or heat-flux).

The state point of the LJ fluid is rho* = 0.6, T* = 1.35, and Rcut =
2.5 sigma.  This was chosen to agree with a 1986 paper by D Evans in
Phys Rev A, 34, p 1449, where he computed the thermal conductivity of
a small 108-atom system using a thermostatting method.  Fig 1 in the
paper shows his simulations produced a kappa of around 3.4 for this
system, in agreement with an experimental data point as well.

-------------

Here is how to extract Kappa from the log file output for each method.

The NEMD methods use the formula kappa = dQ * dZ/dTemp where dQ =
energy flux, and dTemp/dZ = temperature gradient.

(1) in.langevin

dQ = 8000 * 0.5*(0.890+0.883) / 100 / 18.82^2 / 2
  8000 atoms
  0.5*(0.890+0.883) = from log file =
    ave of total in/out energy for 2 regions normalized by # of atoms
  100 = 20,000 steps at 0.005 tau timestep = run time in tau
  xy box area = 18.82^2
  divide by 2 since energy flux goes in 2 directions due to periodic z
dTemp = 0.574 from log file for average Temp difference between 2 regions
dZ = 18.82

Kappa = 3.29

(2) in.heat

dQ = (100*100) / 100 / 18.82^2 / 2
  100*100 = 100 (time in tau) * 100 (energy delta specified in fix heat)
  100 = 20,000 steps at 0.005 tau timestep = run time in tau
  xy box area = 18.82^2
  divide by 2 since energy flux goes in 2 directions due to periodic z
dTemp = 0.748 from log file for average Temp difference between 2 regions
dZ = 18.82

Kappa = 3.55

(3) in.ehex

dQ = (100*100) / 100 / 18.82^2 / 2
  100*100 = 100 (time in tau) * 100 (energy delta specified in fix heat)
  100 = 20,000 steps at 0.005 tau timestep = run time in tau
  xy box area = 18.82^2
  divide by 2 since energy flux goes in 2 directions due to periodic z
dTemp = 0.770 from log file for average Temp difference between 2 regions
dZ = 18.82

Kappa = 3.45

(4) in.mp

dQ = 15068 / 100 / 18.82^2 / 2
  15068 = cumulative delta energy, tallied by fix thermal/conductivity
  100 = 20,000 steps at 0.005 tau timestep = run time in tau
  xy box area = 18.82^2
  divide by 2 since energy flux goes in 2 directions due to periodic z
dTemp = 1.175 from log file for average Temp difference between 2 regions
dZ = 18.82

Kappa = 3.41

(5) in.heatflux

kappa is computed directly within the script, by performing a time
integration of the formulas discussed on the compute heat/flux doc
page - the resulting value prints at the end of the run and is in the
log file

Kappa = 3.88
```

---

## The five official scripts

*Source: `examples/KAPPA/`, verbatim. The README notes these are illustrative: "No guarantee is made that the systems are fully equilibrated or that the runs are long enough to generate good statistics."*

### `in.langevin` — Method 1 - thermostat two slabs with fix langevin

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

### `in.heat` — Method 2 - add/subtract energy with fix heat

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

### `in.ehex` — Method 3 - enhanced heat exchange

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

### `in.mp` — Method 4 - Muller-Plathe reverse NEMD

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

### `in.heatflux` — Method 5 - Green-Kubo equilibrium

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

---

## Command reference

### `fix langevin` — Langevin reservoirs (method 1)

*Source: `doc/src/fix_langevin.rst`*

**Syntax**

```lammps
fix ID group-ID langevin Tstart Tstop damp seed keyword values ...
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
         *yes* = do thermostat rotational degrees of freedom via the angular velocity
       *scale* values = type ratio

**Examples**

```lammps
fix 3 boundary langevin 1.0 1.0 1000.0 699483
fix 1 all langevin 1.0 1.1 100.0 48279 scale 3 1.5
fix 1 all langevin 1.0 1.1 100.0 48279 angmom 3.333
```

**Description** (first paragraphs)

Apply a Langevin thermostat as described in (Bruenger)
to a group of atoms which models an interaction with a background
implicit solvent.  Used with `fix nve`, this command
performs Brownian dynamics (BD), since the total force on each atom
will have the form:

F = & F_c + F_f + F_r \\
   F_f = & - \frac{m}{\mathrm{damp}} v \\
   F_r \propto & \sqrt{\frac{k_B T m}{dt~\mathrm{damp}}}

$F_c$ is the conservative force computed via the usual
inter-particle interactions (`pair_style`,
`bond_style`, etc).  The $F_f$ and $F_r$
terms are added by this fix on a per-particle basis.  See the
`pair_style dpd/tstat` command for a thermostatting
option that adds similar terms on a pairwise basis to pairs of
interacting particles.

**Constraints and cautions from this page**

- **NOTE:** Unlike the `fix nvt` command which performs Nose/Hoover thermostatting AND time integration, this fix does NOT perform time integration.  It only modifies forces to effect thermostatting.  Thus you must use a separate time integration fix, like `fix nve` to actually update the velocities and positions of atoms using the modified forces.  Likewise, this fix should not normally be used on atoms that also have their temperature controlled by another fix - e.g. by `fix nvt` or `fix temp/rescale` commands.

### `fix heat` — add/subtract energy (method 2)

*Source: `doc/src/fix_heat.rst`*

**Syntax**

```lammps
fix ID group-ID heat N eflux
```

* ID, group-ID are documented in `fix` command
* heat = style name of this fix command
* N = add/subtract heat every this many timesteps
* eflux = rate of heat addition or subtraction (energy/time units)
* eflux can be a variable (see below)
* zero or more keyword/value pairs may be appended to args
* keyword = *region*
       *region* value = region-ID

**Examples**

```lammps
fix 3 qin heat 1 1.0
fix 3 qin heat 10 v_flux
fix 4 qout heat 1 -1.0 region top
```

**Description** (first paragraphs)

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

Heat addition/subtraction is performed every N timesteps.

**Constraints and cautions from this page**

- **NOTE:** The heat-exchange (HEX) algorithm implemented by this fix is known to exhibit a pronounced energy drift. An improved algorithm (eHEX) is available as a `fix ehex` command and might be preferable if energy conservation is important.
- **NOTE:** If heat is subtracted from the system too aggressively so that the group's kinetic energy would go to zero, or any individual atom's kinetic energy would go to zero for the case where *eflux* is an atom-style variable, then LAMMPS will halt with an error message.

### `fix ehex` — enhanced heat exchange (method 3)

*Source: `doc/src/fix_ehex.rst`*

**Syntax**

```lammps
fix ID group-ID ehex nevery F keyword value
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

**Examples**

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

**Description** (first paragraphs)

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

\dot{\mathbf r}_i &= \mathbf v_i,  \\
     \dot{\mathbf v}_i &= \frac{\mathbf f_i}{m_i} + \frac{\mathbf g_i}{m_i}.

**Constraints and cautions from this page**

- **NOTE:** This fix only integrates the thermostatting force and must be combined with another integrator, such as `fix nve`, to solve the full equations of motion.
- **NOTE:** If heat is subtracted from the system too aggressively so that the group's kinetic energy would go to zero, then LAMMPS will halt with an error message.  Increasing the value of *nevery* means that heat is added/subtracted less frequently but in larger portions.  The resulting temperature profile will therefore be the same.
- **NOTE:** You can only use the keyword *com* along with *constrain*.
- **NOTE:** Even if RATTLE is used and the keywords *com* and *constrain* are both set, the coordinate constraints will not necessarily be satisfied up to the target precision. The velocity constraints are satisfied as long as all sites of a cluster are rescaled (keyword *com*) and the cluster does not span adjacent reservoirs. The current implementation of the eHEX algorithm introduces a small error in the bond distances, which goes to zero with order three in the timestep. For example, in a simulation of SPC/E water with a timestep of 2 fs the maximum relative error in the bond distances was found to be on the order of $10^{-7}$ for relatively large temperature gradients.  A higher precision can be achieved by decreasing the timestep.

### `fix thermal/conductivity` — Muller-Plathe reverse NEMD (method 4)

*Source: `doc/src/fix_thermal_conductivity.rst`*

**Syntax**

```lammps
fix ID group-ID thermal/conductivity N edim Nbin keyword value ...
```

* ID, group-ID are documented in `fix` command
* thermal/conductivity = style name of this fix command
* N = perform kinetic energy exchange every N steps
* edim = *x* or *y* or *z* = direction of kinetic energy transfer
* Nbin = # of layers in edim direction (must be even number)
* zero or more keyword/value pairs may be appended
* keyword = *swap*
       *swap* value = Nswap = number of swaps to perform every N steps

**Examples**

```lammps
fix 1 all thermal/conductivity 100 z 20
fix 1 all thermal/conductivity 50 z 20 swap 2
```

**Description** (first paragraphs)

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

The simulation box is divided into *Nbin* layers in the *edim*
direction, where the layer 1 is at the low end of that dimension and
the layer *Nbin* is at the high end.  Every N steps, Nswap pairs of
atoms are chosen in the following manner.  Only atoms in the fix group
are considered.  The hottest Nswap atoms in layer 1 are selected.
Similarly, the coldest Nswap atoms in the "middle" layer (see below)
are selected.  The two sets of Nswap atoms are paired up and their
velocities are exchanged.  This effectively swaps their kinetic
energies, assuming their masses are the same.  If the masses are
different, an exchange of velocities relative to center of mass motion
of the two atoms is performed, to conserve kinetic energy.  Over time,
this induces a temperature gradient in the system which can be
measured using commands such as the following, which writes the
temperature profile (assuming z = edim) to the file tmp.profile:

**Constraints and cautions from this page**

- **NOTE:** If your system is periodic in the direction of the heat flux, then the flux is going in 2 directions.  This means the effective heat flux in one direction is reduced by a factor of 2.  You will see this in the equations for thermal conductivity (kappa) in the Muller-Plathe paper.  LAMMPS is simply tallying kinetic energy which does not account for whether or not your system is periodic; you must use the value appropriately to yield a kappa for your system.
- **NOTE:** After equilibration, if the temperature gradient you observe is not linear, then you are likely swapping energy too frequently and are not in a regime of linear response.  In this case you cannot accurately infer a thermal conductivity and should try increasing the Nevery parameter.

### `compute heat/flux` — Green-Kubo (method 5)

*Source: `doc/src/compute_heat_flux.rst`*

**Syntax**

```lammps
compute ID group-ID heat/flux ke-ID pe-ID stress-ID
```

* ID, group-ID are documented in `compute` command
* heat/flux = style name of this compute command
* ke-ID = ID of a compute that calculates per-atom kinetic energy
* pe-ID = ID of a compute that calculates per-atom potential energy
* stress-ID = ID of a compute that calculates per-atom stress

**Examples**

```lammps
compute myFlux all heat/flux myKE myPE myStress
```

**Description** (first paragraphs)

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

The compute takes three arguments which are IDs of other
`computes`.  One calculates per-atom kinetic energy
(*ke-ID*), one calculates per-atom potential energy (*pe-ID)*, and the
third calculates per-atom stress (*stress-ID*).

**Constraints and cautions from this page**

- **NOTE:** These other computes should provide values for all the atoms in the group this compute specifies.  That means the other computes could use the same group as this compute, or they can just use group "all" (or any group whose atoms are superset of the atoms in this compute's group).  LAMMPS does not check for this.
- **NOTE:** The `compute pe/atom` and `compute stress/atom` commands have options for which terms to include in their calculation (pair, bond, etc).  The heat flux calculation will thus include exactly the same terms. Normally you should use `compute stress/atom virial` or `compute centroid/stress/atom virial` so as not to include a kinetic energy term in the heat flux.
- **WARNING:** The compute *heat/flux* has been reported to produce unphysical values for angle, dihedral, improper and constraint force contributions when used with `compute stress/atom`, as discussed in (Surblys2019), (Boone) and (Surblys2021). You are strongly advised to use `compute centroid/stress/atom`, which has been implemented specifically for such cases.
- **WARNING:** Due to an implementation detail, the $y$ and $z$ components of heat flux from `fix rigid` contribution when computed via `compute stress/atom` are highly unphysical and should not be used.

### `compute temp/region` — temperature of a spatial slab

*Source: `doc/src/compute_temp_region.rst`*

**Syntax**

```lammps
compute ID group-ID temp/region region-ID
```

* ID, group-ID are documented in `compute` command
* temp/region = style name of this compute command
* region-ID = ID of region to use for choosing atoms

**Examples**

```lammps
compute mine flow temp/region boundary
```

**Description** (first paragraphs)

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

The temperature is calculated by the formula

### `compute ke/atom` — per-atom KE for profiles

*Source: `doc/src/compute_ke_atom.rst`*

**Syntax**

```lammps
compute ID group-ID ke/atom
```

* ID, group-ID are documented in `compute` command
* ke/atom = style name of this compute command

**Examples**

```lammps
compute 1 all ke/atom
```

**Description** (first paragraphs)

Define a computation that calculates the per-atom translational
kinetic energy for each atom in a group.

The kinetic energy is simply $\frac12 m v^2$, where $m$ is the mass
and $v$ is the velocity of each atom.

The value of the kinetic energy will be 0.0 for atoms not in the
specified compute group.

### `compute chunk/atom` — binning along an axis

*Source: `doc/src/compute_chunk_atom.rst`*

**Syntax**

```lammps
compute ID group-ID chunk/atom style args keyword values ...
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
       *nchunk* value = *once* or *every*
       *limit* values = 0 or Nc max or Nc exact

**Examples**

```lammps
compute 1 all chunk/atom type
compute 1 all chunk/atom bin/1d z lower 0.02 units reduced
compute 1 all chunk/atom bin/2d z lower 1.0 y 0.0 2.5
compute 1 all chunk/atom molecule region sphere nchunk once ids once compress yes
compute 1 all chunk/atom bin/sphere 5 5 5 2.0 5.0 5 discard yes
compute 1 all chunk/atom bin/cylinder z lower 2 10 10 2.0 5.0 3 discard yes
compute 1 all chunk/atom c_cluster
```

**Description** (first paragraphs)

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

Conceptually it is important to realize that this compute does two
simple things.  First, it sets the value of *Nchunk* = the number of
chunks, which can be a constant value or change over time.  Second, it
assigns each atom to a chunk via a chunk ID.  Chunk IDs range from 1
to *Nchunk* inclusive; some chunks may have no atoms assigned to them.
Atoms that do not belong to any chunk are assigned a value of 0.  Note
that the two operations are not always performed together.  For
example, spatial bins can be setup once (which sets *Nchunk*), and
atoms assigned to those bins many times thereafter (setting their
chunk IDs).

**Constraints and cautions from this page**

- **NOTE:** If you want the persistent chunk-IDs calculated by this compute to be continuous when running from a `restart file`, then you should use the same ID for this compute, as in the original run.  This is so that the fix this compute creates to store per-atom quantities will also have the same ID, and thus be initialized correctly with chunk IDs from the restart file.
- **NOTE:** The compression operation requires global communication across all processors to share their chunk ID values.  It can require large memory on every processor to store them, even after they are compressed, if there are a large number of unique chunk IDs with atoms assigned to them.  It uses a STL map to find unique chunk IDs and store them in sorted order.  Each time an atom is assigned a compressed chunk ID, it must access the STL map.  All of this means that compression can be expensive, both in memory and CPU time.  The use of the *limit* keyword in conjunction with the *compress* keyword can affect these costs, depending on which keyword is used first.  So use this option with care.

### `fix ave/chunk` — temperature profile

*Source: `doc/src/fix_ave_chunk.rst`*

**Syntax**

```lammps
fix ID group-ID ave/chunk Nevery Nrepeat Nfreq chunkID value1 value2 ... keyword args ...
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
       *bias* arg = bias-ID
       *adof* value = dof_per_atom

**Examples**

```lammps
fix 1 all ave/chunk 10000 1 10000 binchunk c_myCentro title1 "My output values"
fix 1 flow ave/chunk 100 10 1000 molchunk vx vz norm sample file vel.profile
fix 1 flow ave/chunk 100 5 1000 binchunk density/mass ave running
fix 1 flow ave/chunk 100 5 1000 binchunk density/mass ave running
```

**Description** (first paragraphs)

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

Note that if the `compute chunk/atom`
command defines spatial bins, the fix ave/chunk command performs a
similar computation as the `fix ave/grid` command.
However, the per-bin outputs from the fix ave/chunk command are
global; each processor stores a copy of the entire set of bin data.
By contrast, the `fix ave/grid` command uses a
distributed grid where each processor owns a subset of the bins.  Thus
it is more efficient to use the `fix ave/grid`
command when the grid is large and a simulation is run on many
processors.

**Constraints and cautions from this page**

- **NOTE:** .. versionchanged:: 31May2016 If you are trying to replace a deprecated fix ave/spatial command with the newer, more flexible fix ave/chunk and `compute chunk/atom` commands, you simply need to split the fix ave/spatial arguments across the two new commands.  For example, this command: .. code-block:: LAMMPS fix 1 flow ave/spatial 100 10 1000 y 0.0 1.0 vx vz norm sample file vel.profile could be replaced by: .. code-block:: LAMMPS compute cc1 flow chunk/atom bin/1d y 0.0 1.0 fix 1 flow ave/chunk 100 10 1000 cc1 vx vz norm sample file vel.profile
- **NOTE:** This fix works by creating an array of size $N_\text{chunk} \times N_\text{values}$ on each processor. $N_\text{chunk}$ is the number of chunks, which is defined by the `compute chunk/atom` command. $N_\text{values}$ is the number of input values specified. Each processor loops over its atoms, tallying its values to the appropriate chunk.  Then the entire array is summed across all processors.  This means that using a large number of chunks will incur an overhead in memory and computational cost (summing across processors), so be careful to define a reasonable number of chunks.
- **NOTE:** To perform per-chunk averaging within a $N_\text{freq}$ time window, the number of chunks $N_\text{chunk}$ defined by the `compute chunk/atom` command must remain constant.  If the *ave* keyword is set to *running* or *window* then $N_\text{chunk}$ must remain constant for the duration of the simulation.  This fix forces the chunk/atom compute specified by chunkID to hold $N_\text{chunk}$ constant for the appropriate time windows, by not allowing it to re-calculate $N_\text{chunk}$, which can also affect how it assigns chunk IDs to atoms.  This is particularly important to understand if the chunks defined by the `compute chunk/atom` command are spatial bins.  If its *units* keyword is set to *box* or *lattice*, then the number of bins $N_\text{chunk}$ and size of each bin will be fixed over the $N_\text{freq}$ time window, which can affect which atoms are discarded if the simulation box size changes.  If its *units* keyword is set to *reduced*, then the number of bins $N_\text{chunk}$ will still be fixed, but the size of each bin can vary at each time step if the simulation box size changes (e.g., for an NPT simulation).

### `fix ave/time` — running averages

*Source: `doc/src/fix_ave_time.rst`*

**Syntax**

```lammps
fix ID group-ID ave/time Nevery Nrepeat Nfreq value1 value2 ... keyword args ...
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
       *off* arg = M = do not average this value
       *file* arg = filename

**Examples**

```lammps
fix 1 all ave/time 100 5 1000 c_myTemp c_thermo_temp file temp.profile
fix 1 all ave/time 100 5 1000 c_thermo_press[2] ave window 20 &
                              title1 "My output values"
fix 1 all ave/time 100 5 1000 c_thermo_press[*]
fix 1 all ave/time 1 100 1000 f_indent f_indent[1] file temp.indent off 1
```

**Description** (first paragraphs)

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

Each listed value can be the result of a `compute` or
`fix` or the evaluation of an equal-style or vector-style
`variable`.  In each case, the compute, fix, or variable
must produce a global quantity, not a per-atom or local quantity.  If
you wish to spatial- or time-average or histogram per-atom quantities
from a compute, fix, or variable, then see the
`fix ave/chunk`, `fix ave/atom`,
or `fix ave/histo` commands.  If you wish to sum a
per-atom quantity into a single global quantity, see the
`compute reduce` command.

**Constraints and cautions from this page**

- **NOTE:** For a vector-style variable, only the wildcard forms "\*n" or "m\*n" are allowed.  You must specify the upper bound, because vector-style variable lengths are not determined until the variable is evaluated.  If n is specified larger than the vector length turns out to be, zeroes are output for missing vector values.

### `fix ave/correlate` — autocorrelation for Green-Kubo

*Source: `doc/src/fix_ave_correlate.rst`*

**Syntax**

```lammps
fix ID group-ID ave/correlate Nevery Nrepeat Nfreq value1 value2 ... keyword args ...
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
       *prefactor* args = value
       *file* arg = filename

**Examples**

```lammps
fix 1 all ave/correlate 5 100 1000 c_myTemp file temp.correlate
fix 1 all ave/correlate 1 50 10000 &
          c_thermo_press[1] c_thermo_press[2] c_thermo_press[3] &
          type upper ave running title1 "My correlation data"
fix 1 all ave/correlate 1 50 10000 c_thermo_press[*]
```

**Description** (first paragraphs)

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

Each listed value can be the result of a `compute` or
`fix` or the evaluation of an equal-style or vector-style
`variable`.  In each case, the compute, fix, or variable
must produce a global quantity, not a per-atom or local quantity.  If
you wish to spatial- or time-average or histogram per-atom quantities
from a compute, fix, or variable, then see the
`fix ave/chunk`, `fix ave/atom`, or
`fix ave/histo` commands.  If you wish to convert a
per-atom quantity into a single global value, see the
`compute reduce` command.

**Constraints and cautions from this page**

- **NOTE:** For a vector-style variable, only the wildcard forms "\*n" or "m\*n" are allowed.  You must specify the upper bound, because vector-style variable lengths are not determined until the variable is evaluated.  If n is specified larger than the vector length turns out to be, zeroes are output for missing vector values.

---
