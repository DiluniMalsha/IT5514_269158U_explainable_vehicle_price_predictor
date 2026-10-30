"""Trim-flag extraction from the free-text `edition` field.

The four `trim_*` flags and `trim_grade` are not typed by a seller as fields;
they are tokens pulled out of free text. The front-end has to derive them the
same way the training data did, or a user typing "AMG Line" would get a
prediction from a model that never saw the performance flag set.

The patterns below are a verbatim mirror of the block in
`code/preprocessing/clean_data.py` that writes `trim_stated` through
`trim_equipment`. They are duplicated rather than imported because that script's
output is what the report certifies as reproducible, and refactoring it after the
dataset was frozen risks changing that output. If the cleaner's patterns ever
change, change them here too.
"""

from __future__ import annotations

import re

# Japanese-market grade letters, ordered roughly base -> mid -> high.
_GRADE_IN = re.compile(r'\b([gxz])\s*grade\b')
_GRADE_AT_START = re.compile(r'^([gxz])\b')

_PERFORMANCE = re.compile(
    r'\b(?:amg|gtx?|rs|sport|turbo|nismo|type\s*r|m\s*sport|st[- ]line)\b')
_PREMIUM = re.compile(
    r'\b(?:premium|luxury|highest|fully\s*loaded|autobiography|limited|exclusive)\b')
_EQUIPMENT = re.compile(
    r'\b(?:safety|bsm|led|4\s*way|360|camera|push\s*start|sunroof)\b')


def extract(edition: str | None) -> dict:
    """Return the five trim features for one free-text edition string.

    An empty box means the seller stated no trim, which is what `trim_stated`
    records; it is a real signal in the data rather than a missing value.
    """
    text = (edition or '').strip()
    stated = bool(text) and text.lower() != 'unknown'
    low = re.sub(r'\s+', ' ', text).lower()

    grade_match = _GRADE_IN.search(low) or _GRADE_AT_START.search(low)
    grade = grade_match.group(1).upper() if grade_match else 'none'

    return {
        'trim_stated': int(stated),
        'trim_grade': grade,
        'trim_performance': int(bool(_PERFORMANCE.search(low))),
        'trim_premium': int(bool(_PREMIUM.search(low))),
        'trim_equipment': int(bool(_EQUIPMENT.search(low))),
    }


# Shown in the interface so a user can see why a flag did or did not fire,
# rather than being told a number they cannot trace.
LABELS = {
    'trim_performance': 'Performance line (AMG, GT, RS, Sport, Turbo, Nismo, Type R, M Sport, ST-Line)',
    'trim_premium': 'Premium wording (Premium, Luxury, Highest, Fully Loaded, Autobiography, Limited, Exclusive)',
    'trim_equipment': 'Equipment package (Safety, BSM, LED, 4-way, 360, Camera, Push Start, Sunroof)',
}
