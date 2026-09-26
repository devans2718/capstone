"""Every registered seed, in one place (pre-registration §3, *Registered seeds*).

WHY A MODULE AND NOT ARGUMENTS
    A seed that a script takes on its command line is a seed that can be
    changed after a result without editing anything a reviewer reads. The
    values below are the ones revision 3 registered by value before any run;
    the probe harness reads them from here and offers no flag to override
    them. Changing one is a §7 deviation, not a configuration choice.

    FOLDS and SUBSAMPLE share a value, as registered, but never a stream: the
    folds come from scikit-learn's StratifiedKFold, the subsample from a numpy
    generator keyed on (seed, fold) inside capstone.splits.
"""

FOLDS = 20260926         # random's StratifiedKFold(shuffle=True)
SUBSAMPLE = 20260926     # the secondary run's subsample, keyed on (seed, fold)
CONTROL = 20260927       # the control task's type key -> label mapping
BOOTSTRAP = 20260928     # the word-type bootstrap: criterion 1 and per-layer CIs
CONTROL_NULL = 20260929  # criterion 2's 10,000 null mappings

# The demo slice's synthetic draw. Not part of the registered design.
SYNTHETIC = 20260919

REGISTERED = {
    "folds": FOLDS,
    "subsample": SUBSAMPLE,
    "control": CONTROL,
    "bootstrap": BOOTSTRAP,
    "control_null": CONTROL_NULL,
}
