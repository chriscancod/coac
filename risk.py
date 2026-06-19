"""
Risk scoring engine.

Every action is scored 0.0–1.0 before execution.
Score < threshold (0.40 in COFOUNDER mode) → execute autonomously.
Score >= threshold → push interrupt, halt, await founder approval.

Scores are additive and capped at 1.0. Each factor contributes
a weighted sub-score so the dominant risk dimension is always visible
in the breakdown returned alongside the total.
"""

from dataclasses import dataclass, field
from typing import Literal


ActionCategory = Literal[
    "file_edit",
    "git_commit",
    "git_push",
    "railway_deploy",
    "cad_overwrite",
    "cad_export",
    "brand_listing",
    "brand_sale",
    "db_write",
    "db_delete",
    "notification_send",
    "read_only",
]

# Blast-radius tiers — how wide the blast radius is if the action goes wrong.
BLAST_RADIUS_SCORES: dict[str, float] = {
    "single_file":       0.05,
    "multiple_files":    0.10,
    "repo":              0.20,
    "production_system": 0.35,
    "customer_data":     0.40,
}

# Irreversibility penalties — how hard it is to undo.
IRREVERSIBILITY_SCORES: dict[ActionCategory, float] = {
    "read_only":         0.00,
    "file_edit":         0.05,
    "git_commit":        0.08,
    "db_write":          0.10,
    "cad_export":        0.10,
    "notification_send": 0.12,
    "git_push":          0.20,
    "brand_listing":     0.20,
    "cad_overwrite":     0.25,
    "railway_deploy":    0.30,
    "brand_sale":        0.35,
    "db_delete":         0.45,
}

# External side-effect premium — applies when the action leaves the local machine.
EXTERNAL_EFFECT_SCORES: dict[ActionCategory, float] = {
    "read_only":         0.00,
    "file_edit":         0.00,
    "git_commit":        0.00,
    "cad_overwrite":     0.00,
    "cad_export":        0.00,
    "db_write":          0.05,
    "db_delete":         0.10,
    "notification_send": 0.05,
    "git_push":          0.15,
    "railway_deploy":    0.20,
    "brand_listing":     0.15,
    "brand_sale":        0.25,
}


@dataclass
class RiskScore:
    total: float
    category: ActionCategory
    blast_radius_tier: str
    irreversibility: float
    external_effect: float
    data_sensitivity_penalty: float
    breakdown: dict[str, float] = field(default_factory=dict)
    requires_approval: bool = False
    rationale: str = ""

    def __post_init__(self):
        self.total = min(self.total, 1.0)
        self.requires_approval = self.total >= 0.40
        self.breakdown = {
            "irreversibility":        self.irreversibility,
            "blast_radius":           BLAST_RADIUS_SCORES.get(self.blast_radius_tier, 0.0),
            "external_effect":        self.external_effect,
            "data_sensitivity":       self.data_sensitivity_penalty,
        }


def score(
    category: ActionCategory,
    blast_radius_tier: str = "single_file",
    touches_customer_data: bool = False,
    touches_payment_flow: bool = False,
    custom_penalty: float = 0.0,
) -> RiskScore:
    """
    Score an action before execution.

    Args:
        category: What kind of action this is.
        blast_radius_tier: Scope of damage if it fails
            ('single_file', 'multiple_files', 'repo',
             'production_system', 'customer_data').
        touches_customer_data: True if the action reads/writes PII or
            customer records — adds a 0.20 data sensitivity penalty.
        touches_payment_flow: True if the action touches payment
            processing — adds a 0.30 penalty on top of everything else.
        custom_penalty: Caller-supplied override for edge cases.

    Returns:
        RiskScore with total, breakdown, and requires_approval flag.
    """
    irreversibility = IRREVERSIBILITY_SCORES.get(category, 0.10)
    blast            = BLAST_RADIUS_SCORES.get(blast_radius_tier, 0.05)
    external         = EXTERNAL_EFFECT_SCORES.get(category, 0.00)

    data_sensitivity = 0.0
    if touches_customer_data:
        data_sensitivity += 0.20
    if touches_payment_flow:
        data_sensitivity += 0.30

    total = irreversibility + blast + external + data_sensitivity + custom_penalty

    rationale_parts = []
    if irreversibility >= 0.25:
        rationale_parts.append("high irreversibility")
    if blast >= 0.20:
        rationale_parts.append("wide blast radius")
    if external >= 0.15:
        rationale_parts.append("external side-effects")
    if data_sensitivity > 0:
        rationale_parts.append("sensitive data involved")

    rationale = (
        "Requires approval: " + ", ".join(rationale_parts)
        if rationale_parts
        else "Within autonomous execution threshold."
    )

    return RiskScore(
        total=round(min(total, 1.0), 3),
        category=category,
        blast_radius_tier=blast_radius_tier,
        irreversibility=irreversibility,
        external_effect=external,
        data_sensitivity_penalty=data_sensitivity,
        rationale=rationale,
    )


def score_from_dict(params: dict) -> RiskScore:
    """Convenience wrapper for tool calls that pass kwargs as a dict."""
    return score(**params)
