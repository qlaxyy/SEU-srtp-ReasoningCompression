"""New borrowed lexical mechanism, not an official CGRS implementation.

Fixed before new generation: subtract log(2) from reflection logits only at a
clean step opening, optionally requiring the existing ReBalance coefficient <0.
No probe, answer substitution, extra RNG, or claim that confidence proves truth.
"""
import math

PENALTY = math.log(2)


def validate_penalty(mode, lower_bound, constant_scale):
    if mode == 'fixed':
        if lower_bound is not None or constant_scale is not None:
            raise ValueError('Fixed penalty takes no new parameters')
        return
    if mode not in ('coefficient_scaled', 'calibration_constant'):
        raise ValueError('Unknown penalty mode')
    if lower_bound is None or not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError('Finite negative model-specific lower bound required')
    if mode == 'calibration_constant':
        if constant_scale is None or not math.isfinite(constant_scale) or not 0 < constant_scale <= 1:
            raise ValueError('Frozen calibration scale must be in (0, 1]')
    elif constant_scale is not None:
        raise ValueError('Scaled mode takes no constant scale')


def device_penalty(torch, coefficient, mode, lower_bound, constant_scale):
    """New modes only; fixed mode retains its original scalar subtraction."""
    if mode == 'calibration_constant':
        return torch.full_like(coefficient, PENALTY * constant_scale)
    if mode != 'coefficient_scaled':
        raise ValueError(mode)
    safe = torch.where(torch.isfinite(coefficient), coefficient, torch.zeros_like(coefficient))
    return PENALTY * torch.clamp(safe / lower_bound, min=0.0, max=1.0)
TRIGGERS = {
    14190: 'Wait', 13824: ' Wait', 11489: 'wait', 3783: ' wait',
    3983: 'But', 1988: ' But', 8088: 'but', 714: ' but',
    92014: 'Alternatively', 38478: ' Alternatively',
    75763: 'Alternative', 41109: ' Alternative',
    80022: 'Hmm', 88190: ' Hmm',
}

# Fixed semantic ablation: preserve Wait/Alternatively/Hmm, release generic
# contrast and adjectival Alternative. A semantic hypothesis, not a validated rule.
NARROW_EXCLUDED = frozenset((3983, 1988, 8088, 714, 75763, 41109))


def trigger_vocabulary(profile='original14'):
    if profile == 'original14':
        return dict(TRIGGERS)
    if profile == 'narrow8':
        return {token: piece for token, piece in TRIGGERS.items()
                if token not in NARROW_EXCLUDED}
    raise ValueError('Unknown trigger profile: ' + str(profile))


def token_flags(piece, rebalance_boundary):
    """Mixed boundary/content tokens must not open a lexical window."""
    opening = rebalance_boundary and not piece.rsplit('\n\n', 1)[-1].strip()
    return opening, not piece.strip()


def eligible(opening, thinking, coefficient, previous_mean, mode):
    if mode not in ('off', 'shadow', 'negative', 'always'):
        raise ValueError(mode)
    if mode == 'off' or not opening or not thinking:
        return False
    if mode == 'always':
        return True
    return math.isfinite(previous_mean) and math.isfinite(coefficient) and coefficient < 0


def next_opening(was_open, piece, boundary, token_id):
    if token_id in (151648, 151649):
        return False
    clean, whitespace = token_flags(piece, boundary)
    return clean if boundary else was_open and whitespace
