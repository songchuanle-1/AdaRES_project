# Resume Probe Optimizer Config Mutation

Lease a1526671b2ae48f7a80c4f8128bf645a, 2026-09-30.
Third probe reached native resume but optimizer.load_state_dict raised
"different number of parameter groups".
Inspected installed gorilla/solver/build.py: builder mutates its input config,
and scheduler attaches its optimizer to the config. The probe reused those
mutated configs to construct its second instance. Corrected both builders to
use fresh deep copies. Production tools/train.py constructs once per process;
this probe failure does not establish a production optimizer resume defect.
No optimizer state is dropped or silently reset.
