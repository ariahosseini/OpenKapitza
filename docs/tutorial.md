# Tutorial: from a joint to an answer you can defend (15 minutes)

Five short steps, each with the real output of the commands shown (version
0.2.0). Install first: [`install.md`](install.md) (the Python part is
enough until step 5).

## 1. One joint, from Python

Two stainless-steel faces, bead-blasted to σ = 1.2 µm, pressed at 2 MPa in
air:

```python
from openkapitza import build_joint, run_continuum, render

joint = build_joint("SS304", "SS304", sigma_a_um=1.2, sigma_b_um=1.2,
                    slope_a=0.08, slope_b=0.08, pressure_MPa=2.0,
                    temperature_K=300.0, gas="air", gas_pressure_kPa=101.325)
print(render(run_continuum(joint)))
```

The report lists every intermediate with its source. The lines to look at
first:

```
  P/H_c                       0.0005868
  psi (plasticity index)          3.497   -> plastic
  h_g                              4650 W/m^2.K
  h_solid (series)                 1147 W/m^2.K
  h_JOINT                          5796 W/m^2.K   [low]
    gas gap (parallel)               80.22 %   share of total conductance
  would MD change this?
    h_K swept by a factor of 2.50 moves h_joint by 0.1 %
```

- The real contact area is 0.06 % of the face (P/H_c), and the asperities
  deform plastically, so the CMY model applies.
- **The air in the gaps carries 80 % of the heat.** Evacuate the joint and
  the conductance falls about 5×.
- The interface (Kapitza) term is irrelevant here: changing it 2.5× moves
  the answer 0.1 %. No molecular dynamics needed.
- `[low]` is the confidence. It is low because the interface term is a
  placeholder and some inputs are defaults, and the report says which.

## 2. A real assembly, from a description file

A heat sink screwed to an aluminium lid, written as YAML
([`examples/agent/heatsink_screwed.yaml`](../examples/agent/heatsink_screwed.yaml)):

```bash
python3 -m openkapitza.agent --input examples/agent/heatsink_screwed.yaml
```

```
Joint conductance: 6.37 kW/m²K  (resistance 157 mm²K/W)
  90 % band from the input uncertainties: 5 kW/m²K – 8.21 kW/m²K
Heat splits: 49% through the solid contact, 51% through the gas.
Resistance of the solid path, by part:
  - macro constriction (flatness): 95%
  - constriction (contact spots): 5%
Atomistic (MD) calculation: not needed — sweeping the interface term moves the answer by 0.5%
What to watch:
  [WARNING] Out-of-flatness limits the contact to the central 30% of the joint radius ...
Best next measurements:
  - Measure h on the assembled, bolted joint, or at least the face flatness and the pressure map
  - Lap the faces flatter or raise the load; or fill the macro gap with a TIM.
```

Read it in this order: the number and its band; **where the resistance
is** (here the faces are not flat enough, and the microscopic roughness
hardly matters); whether MD would change anything; the warnings; and what
to measure next to shrink the band. Every warning has a stable code
(`--json -` prints them all, with the inputs and defaults used).

## 3. What if?

```bash
python3 -m openkapitza.agent --input examples/agent/heatsink_screwed.yaml \
    --sweep pressure_MPa=0.1:5:5:log
```

```
  pressure_MPa [MPa]     h [W/m²K]   R [mm²K/W]  limiting part
                 0.1          4681        213.6  macro constriction (flatness)
              0.7071          6964        143.6  macro constriction (flatness)
                   5     1.557e+04        64.22  macro constriction (flatness)
```

Fifty times the load gives about three times the conductance, and flatness
limits it all the way. Add `--csv out.csv` or `--plot out.png` (needs
`pip install -e ".[plots]"`). Any field can be swept; `--fields` lists them.

## 4. When the interface does matter

A polished silicon die on aluminium in vacuum, with a finished MD run of
the Si/Al interface folded in
([`examples/agent/si_al_contact_md.yaml`](../examples/agent/si_al_contact_md.yaml)):

```bash
python3 -m openkapitza.agent --input examples/agent/si_al_contact_md.yaml
```

```
Joint conductance: 905 kW/m²K  (resistance 1.1 mm²K/W)
Where the resistance is:
  - constriction (contact spots): 86%
  - Kapitza (interface, area-scaled): 14%
Atomistic (MD) calculation: in use — h_K = 444.6 ± 173 MW/m²K ...
  [WARNING] T = 300 K is 0.47 of the Debye temperature of 'Si' ... would be 357-402 (a bracket, not a correction).
  [note] The MD h_K (445 MW/m²K) is 1.17x the highest measured value, inside the 1.25x tolerance ...
```

Now the interface is 14 % of the resistance, so its value matters. The MD
result comes with its own error, the classical-MD quantum bracket, and a
comparison with 40 measured Si/Al values from 11 papers. The run here is a
small "plumbing" test, and the warnings say so (39 % error, one cell size).

## 5. Running the MD yourself

Needs LAMMPS ([`install.md`](install.md#2-lammps-for-molecular-dynamics)).
In the guided session:

```bash
python3 -m openkapitza.agent --potentials ~/lammps-openkapitza/potentials
```

```
> run                                  # answer the questions, then run
> prepare jobs ~/ok_jobs               # write the MD jobs the plan names
> submit plan local lmp=~/lammps-openkapitza/build/lmp np=4
> watch                                # poll until done; finished runs are measured
```

Finished runs are measured and folded into the answer automatically, as
in step 4. A failed run is diagnosed (with the knowledge-base entry that
explains it), and a run killed by the machine resumes from its last
checkpoint. Details: [agent guide](agent-guide.md#the-md-loop).

## Where next

- Every field, command and output: [`agent-guide.md`](agent-guide.md).
- The equations and how well each part is validated: the
  [README](../README.md#equations).
- From Claude or another LLM, in your own words: the MCP server
  ([agent guide](agent-guide.md)).
