import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GraphSignal:
    symbol: str
    vote: str
    target_position_pct: float
    confidence: float
    reason: str = ""
    raw: Optional[Dict[str, Any]] = None


def build_graph_input(
    symbol: str,
    date_str: str,
    close_price: float,
    cash: float,
    portfolio_value: float,
    shares: int,
    avg_cost: float = 0.0,
) -> str:
    portfolio_context = {
        "cash": float(cash),
        "portfolio_value": float(portfolio_value),
        "holdings": [
            {
                "symbol": symbol,
                "shares": int(shares),
                "avg_cost": float(avg_cost),
            }
        ],
    }
    return (
        f"symbol={symbol}\n"
        f"date={date_str}\n"
        f"close_price={float(close_price)}\n"
        f"portfolio_context={json.dumps(portfolio_context, ensure_ascii=False)}\n"
    )


def clamp_pct(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def map_vote_to_target_pct(vote: str, current_pct: float) -> float:
    vote_upper = (vote or "HOLD").upper()
    if vote_upper == "STRONG_BUY":
        return 0.60
    if vote_upper == "BUY":
        return 0.30
    if vote_upper == "STRONG_SELL":
        return 0.00
    if vote_upper == "SELL":
        return 0.10
    return clamp_pct(current_pct)


def extract_graph_signal(
    graph_result: Dict[str, Any],
    current_position_pct: float,
    symbol_fallback: str,
) -> GraphSignal:
    final_decision = graph_result.get("final_decision") or {}
    vote = (final_decision.get("final_vote") or "HOLD").upper()
    confidence = float(final_decision.get("confidence", 0.0))

    target_position_pct = final_decision.get("target_position_pct")
    if target_position_pct is None:
        target_position_pct = map_vote_to_target_pct(vote, current_position_pct)
    target_position_pct = clamp_pct(float(target_position_pct))

    details = final_decision.get("details") or {}
    morefit_reason = ((details.get("morefit") or {}).get("reason") or "").strip()
    technical_reason = ((details.get("technical") or {}).get("reason") or "").strip()
    reason = " | ".join([r for r in [morefit_reason, technical_reason] if r])

    return GraphSignal(
        symbol=final_decision.get("symbol") or symbol_fallback,
        vote=vote,
        target_position_pct=target_position_pct,
        confidence=confidence,
        reason=reason,
        raw=graph_result,
    )
