"""Bounded monotone penalty curves. No model, CUDA, labels, or online fitting."""
import math

PENALTY = math.log(2.)


def validate(maximum, lower_bound, curve='linear', shape=1.):
    if not math.isfinite(maximum) or not 1. <= maximum <= 32.:
        raise ValueError('Reviewed multiplier range is [1,32]')
    if not math.isfinite(lower_bound) or lower_bound >= 0:
        raise ValueError('Finite negative native bound required')
    if curve not in ('linear', 'rational'):
        raise ValueError('Unknown curve')
    if not math.isfinite(shape) or not 1e-4 <= shape <= 1e4:
        raise ValueError('Finite, nondegenerate shape required')
    if curve == 'linear' and shape != 1.:
        raise ValueError('Linear control has no fitted shape')


def multiplier(torch, coefficients, maximum, lower_bound, curve='linear', shape=1.):
    safe = torch.where(torch.isfinite(coefficients), coefficients,
                       torch.zeros_like(coefficients)).float()
    s = torch.clamp(safe / lower_bound, min=0., max=1.)
    if curve == 'rational':
        # rho > 0: f(0)=0, f(1)=1, derivative rho/(rho+(1-rho)*s)^2 > 0.
        s = s / (s + shape * (1. - s))
    return 1. + (maximum - 1.) * s


def adjusted_values(torch, values, coefficients, maximum, lower_bound, curve='linear', shape=1.):
    k = multiplier(torch, coefficients, maximum, lower_bound, curve, shape)
    original = values - PENALTY
    if maximum == 1.:
        return original, k
    proposal = (values.float() - (PENALTY * k)[:, None]).to(values.dtype)
    # Preserve original and fixed-endpoint BF16 arithmetic, including infinities.
    proposal = torch.where((k == 1.)[:, None], original, proposal)
    proposal = torch.where((k == maximum)[:, None], values - PENALTY * maximum, proposal)
    return proposal, k
