"""
decision.py - Tầng 3: Confidence Filter + Kelly Criterion + Bet Sizing
"""
import math


def confidence_filter(probs: dict, threshold: float = 0.55) -> dict:
    """
    Chỉ recommend nếu max probability >= threshold.
    Returns {recommend: bool, color: str, confidence: float, skip_reason: str|None}
    """
    best_color = max(probs, key=probs.get)
    confidence = probs[best_color]

    if confidence >= threshold:
        return {
            'recommend': True,
            'color': best_color,
            'confidence': round(confidence, 4),
            'skip_reason': None
        }
    else:
        return {
            'recommend': False,
            'color': best_color,
            'confidence': round(confidence, 4),
            'skip_reason': f'Confidence {confidence:.1%} < threshold {threshold:.1%}'
        }


def kelly_criterion(p_win: float, odds: float, multiplier: float = 0.5) -> float:
    """
    Kelly Criterion: f* = (p * (b+1) - 1) / b
    p_win: probability of winning
    odds: net payout (e.g. T/CT = 1.0, Bonus = 13.0)
    multiplier: Risk reduction factor (e.g. 0.25, 0.5, 1.0)
    Returns: fraction of bankroll to bet (0 ~ 0.10)
    """
    if odds <= 0 or p_win <= 0:
        return 0.0

    f = (p_win * (odds + 1) - 1) / odds

    # Hard cap at 10% of bankroll for stability
    f = max(0.0, min(f, 0.10))

    # Apply risk multiplier (Half-Kelly, Full-Kelly, etc.)
    f *= multiplier

    return round(f, 4)


def bet_size_output(kelly_fraction: float, bankroll: float, min_bet: float = 1.0) -> dict:
    """
    Calculate actual bet amount from Kelly fraction.
    Returns {bet_amount: float, fraction: float, risk_level: str}
    """
    bet = kelly_fraction * bankroll
    bet = max(0, round(bet, 2))

    # Round to min_bet increments
    if bet > 0 and bet < min_bet:
        bet = min_bet

    # Risk level
    pct = bet / bankroll if bankroll > 0 else 0
    if pct > 0.07:
        risk = 'HIGH'
    elif pct > 0.03:
        risk = 'MEDIUM'
    elif pct > 0:
        risk = 'LOW'
    else:
        risk = 'SKIP'

    return {
        'bet_amount': bet,
        'fraction': round(kelly_fraction, 4),
        'risk_level': risk,
        'pct_bankroll': round(pct * 100, 2)
    }


def make_decision(ensemble_result: dict, bankroll: float = 100.0,
                   min_bet: float = 1.0, confidence_threshold: float = 0.55,
                   regime_info: dict = None, kelly_mult: float = 0.5) -> dict:
    """
    Full decision pipeline: ensemble → confidence filter → kelly → bet size.
    Item 8 Fix: Incorporate Regime signals to adjust confidence.
    """
    probs = ensemble_result.get('probs', {'T': 0.33, 'CT': 0.33, 'Bonus': 0.33})
    
    # ─── Item 8: Regime-Aware Adjustment ───
    if regime_info and regime_info.get('regime') != 'STABLE':
        regime = regime_info['regime']
        reg_conf = regime_info.get('confidence', 0.5)
        
        if regime == 'STREAK':
            streak_color = regime_info.get('streak_color')
            if streak_color in probs:
                # Boost probability of the streak continuing
                probs[streak_color] *= (1.0 + (reg_conf * 0.20)) # Up to 20% boost
        
        elif regime == 'ALTERNATING':
            # Boost both T and CT slightly if switch rate is very high
            for c in ['T', 'CT']:
                if c in probs:
                    probs[c] *= (1.0 + (reg_conf * 0.1))

        # Re-normalize after boosting
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

    # Step 1: Confidence filter
    cf = confidence_filter(probs, confidence_threshold)

    if not cf['recommend']:
        return {
            'action': 'SKIP',
            'color': cf['color'],
            'confidence': cf['confidence'],
            'bet_amount': 0,
            'skip_reason': cf['skip_reason'],
            'model_votes': ensemble_result.get('model_votes', {}),
            'probs': probs,
            'regime': regime_info.get('regime', 'STABLE') if regime_info else 'STABLE'
        }

    # Step 2: Kelly criterion
    color = cf['color']
    p_win = probs[color]

    # Odds: T/CT pay 2x (net=1), Bonus pays 14x (net=13)
    if color == 'Bonus':
        odds = 13.0
    else:
        odds = 1.0

    kelly_f = kelly_criterion(p_win, odds, multiplier=kelly_mult)

    # Step 3: Bet sizing
    bet_info = bet_size_output(kelly_f, bankroll, min_bet)

    return {
        'action': 'BET',
        'color': color,
        'confidence': cf['confidence'],
        'bet_amount': bet_info['bet_amount'],
        'fraction': bet_info['fraction'],
        'risk_level': bet_info['risk_level'],
        'pct_bankroll': bet_info['pct_bankroll'],
        'kelly_raw': kelly_f,
        'skip_reason': None,
        'model_votes': ensemble_result.get('model_votes', {}),
        'probs': probs,
        'regime': regime_info.get('regime', 'STABLE') if regime_info else 'STABLE'
    }
