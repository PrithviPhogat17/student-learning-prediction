"""Single source of truth for model inputs and the prediction target.

Every input listed here comes from the UCI "Student Performance" dataset
(Cortez & Silva, 2008). Column meanings follow the UCI variable documentation.

Design choices (explained in the README):
* Prior grades G1/G2 are optional (config: task.include_prior_grades).
* Demographic and sensitive attributes (sex, age, address, family structure,
  parental education/jobs, alcohol use, romantic status, health) are NOT used.
  The model only sees study-related and school-related indicators, so that
  recommendations stay actionable and no one is profiled on personal traits.
* The final grade G3 is never an input; it is only used to build the target.
"""
from __future__ import annotations

from dataclasses import dataclass

TARGET_COLUMN = "G3"          # final grade (0-20), used ONLY to derive the label
LABEL_COLUMN = "at_risk"      # 1 if G3 < pass_mark else 0
RECORD_ID = "record_id"       # anonymous row number
YES_NO = ("no", "yes")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    kind: str                 # "numeric" or "binary"
    group: str
    help: str
    min: float | None = None
    max: float | None = None
    default: float | str = 0
    integer: bool = True
    options: tuple = ()


ALL_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("G1", "Period 1 grade", "numeric", "Earlier grades",
                "First-period grade on the 0-20 scale.", 0, 20, 10),
    FeatureSpec("G2", "Period 2 grade", "numeric", "Earlier grades",
                "Second-period grade on the 0-20 scale.", 0, 20, 10),
    FeatureSpec("studytime", "Weekly study time (1-4)", "numeric", "Study habits",
                "1 = under 2 h, 2 = 2-5 h, 3 = 5-10 h, 4 = over 10 h per week.", 1, 4, 2),
    FeatureSpec("absences", "School absences", "numeric", "Study habits",
                "Number of school absences (0-93 in the dataset).", 0, 93, 0),
    FeatureSpec("failures", "Past class failures", "numeric", "Study habits",
                "Number of past class failures (0 = none).", 0, 4, 0),
    FeatureSpec("traveltime", "Home-to-school travel time (1-4)", "numeric", "Daily context",
                "1 = under 15 min, 2 = 15-30 min, 3 = 30-60 min, 4 = over 1 h.", 1, 4, 1),
    FeatureSpec("freetime", "Free time after school (1-5)", "numeric", "Daily context",
                "1 = very low ... 5 = very high.", 1, 5, 3),
    FeatureSpec("goout", "Going out with friends (1-5)", "numeric", "Daily context",
                "1 = very low ... 5 = very high.", 1, 5, 3),
    FeatureSpec("schoolsup", "Extra school support", "binary", "Support",
                "Does the student receive extra educational support from the school?",
                default="no", options=YES_NO),
    FeatureSpec("famsup", "Family educational support", "binary", "Support",
                "Does the family provide educational support?", default="yes", options=YES_NO),
    FeatureSpec("paid", "Extra paid classes in this subject", "binary", "Support",
                "Attends paid extra classes in the course subject?", default="no", options=YES_NO),
    FeatureSpec("activities", "Extra-curricular activities", "binary", "Daily context",
                "Takes part in extra-curricular activities?", default="yes", options=YES_NO),
    FeatureSpec("higher", "Wants higher education", "binary", "Daily context",
                "Does the student want to pursue higher education?", default="yes", options=YES_NO),
    FeatureSpec("internet", "Internet access at home", "binary", "Daily context",
                "Is there internet access at home?", default="yes", options=YES_NO),
)

PRIOR_GRADE_FEATURES = ("G1", "G2")


def get_feature_specs(include_prior_grades: bool = True) -> list[FeatureSpec]:
    """Feature specs used by the model, in a fixed order."""
    return [s for s in ALL_FEATURES
            if include_prior_grades or s.name not in PRIOR_GRADE_FEATURES]


def spec_by_name(name: str) -> FeatureSpec:
    for spec in ALL_FEATURES:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown feature: {name}")


REQUIRED_COLUMNS: list[str] = [s.name for s in ALL_FEATURES] + [TARGET_COLUMN]
