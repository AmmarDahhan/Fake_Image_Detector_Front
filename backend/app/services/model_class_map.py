"""Model class -> binary API verdict mapping.

==============================================================================
PROVISIONAL - pending confirmation from the AI team.
==============================================================================

The trained model is a 3-class classifier. Its index -> name mapping is read
from the checkpoint itself (``checkpoint["classes"]``) and is therefore not
hardcoded anywhere; the reference run reports it as::

    0 = full_synthetic
    1 = real
    2 = tampered

The public API contract is binary (``real`` | ``fake``). Collapsing three
classes onto two is a *product* decision, not a model fact, so it is recorded
in exactly one place - here - rather than being scattered through the service,
the routes and the tests.

    full_synthetic -> fake
    real           -> real
    tampered       -> fake

``tampered`` is currently folded into ``fake`` on the reading that an edited
image is not authentic. That is an assumption, and it is wrong for the case of
a *manipulated* real photograph: such an image is authentic in origin but
altered, and "fake" conflates "never existed" with "was edited". The AI team
has not yet stated which of those the product should report, so this mapping
must be revisited before the API is treated as settled.

If the AI team confirms a different collapse, change it here and nowhere else.

``PROVISIONAL_BINARY_VERDICT_MAP`` is keyed by the class *name* rather than the
class *index* on purpose: the index order is a property of the checkpoint
(which this module never reads), whereas the name is the stable contract the
AI team documented.
"""

from __future__ import annotations

from app.schemas.analysis import Verdict

# --- The single provisional mapping. Edit here, nowhere else. ---------------
PROVISIONAL_BINARY_VERDICT_MAP: dict[str, Verdict] = {
    "full_synthetic": Verdict.FAKE,
    "real": Verdict.REAL,
    "tampered": Verdict.FAKE,
}

#: Class names the mapping above is known to cover. A checkpoint that reports a
#: name outside this set is a contract change, not something to guess at.
KNOWN_MODEL_CLASSES: frozenset[str] = frozenset(PROVISIONAL_BINARY_VERDICT_MAP)


def verdict_for_class(class_name: str) -> Verdict:
    """Return the binary verdict for a model class name.

    Raises:
        KeyError: the model reported a class the mapping does not cover. This
            is deliberately a hard failure: silently defaulting an unknown
            class to ``fake`` (or ``real``) would make the endpoint answer with
            a verdict nobody chose.
    """
    return PROVISIONAL_BINARY_VERDICT_MAP[class_name]
