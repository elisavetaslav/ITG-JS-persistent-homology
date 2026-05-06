import math
from fractions import Fraction
from functools import reduce


# -----------------------------------------------------------------------
# Duration normalisation:
# - Raw durations are stored as rational numbers in quarter‑length units.
# - To obtain integer durations comparable across pieces, we compute the
#   greatest common divisor (GCD) of all rational durations in the corpus.
#   This GCD is the smallest common time unit (e.g., 1/6 of a quarter).
# - We then take its denominator as the global multiplier: multiplying
#   every raw duration by this denominator yields an integer count of
#   minimal units.
# - This ensures that all rhythmic values are expressed on the same
#   discrete time grid without rounding.
# -----------------------------------------------------------------------


def gcd_rational(numbers):
    """Compute the greatest common divisor of a list of rational numbers."""
    denom_lcm = reduce(lambda a, b: a * b // math.gcd(a, b), [f.denominator for f in numbers])
    numerators = [f.numerator * (denom_lcm // f.denominator) for f in numbers]
    return Fraction(reduce(math.gcd, numerators), denom_lcm)


def compute_global_unit(df):
    """
    Compute global normalisation multiplier for durations.
    INPUT: df - pandas DataFrame with 'parts' column containing per-part dicts with 'durations'.
    RETURN: integer multiplier (denominator of the GCD of all durations).
    """
    all_durs = []
    for _, row in df.iterrows():
        parts = row.get('parts')
        if not parts:
            continue
        for part in parts:
            if part is None:
                continue
            for d in part['durations']:
                all_durs.append(Fraction(str(d)).limit_denominator(512))
    if not all_durs:
        return Fraction(1, 1)
    unit = gcd_rational(all_durs)
    multiplier = unit.denominator
    return multiplier
