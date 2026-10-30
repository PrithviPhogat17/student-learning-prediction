"""Transparent, rule-based study suggestions.

This is NOT the neural network and NOT generative AI. The ANN supplies a risk
probability; these hand-written rules look at the same observed inputs and turn
them into advice. Each suggestion states the rule that fired and the evidence.

The dataset has no topic-level information, so the rules can only talk about
study habits and overall grades, never about specific topics.
"""
from __future__ import annotations

from dataclasses import dataclass

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

LIMITATION_NOTE = (
    "The dataset contains overall period grades and study-habit indicators only. It has no "
    "topic-by-topic results, so these suggestions are general. Ask your teacher which topics "
    "need the most attention."
)


@dataclass
class Recommendation:
    rule_id: str
    title: str
    advice: str
    evidence: str
    priority: str   # "high" | "medium" | "low"


def generate_recommendations(inputs: dict, probability: float | None = None,
                             pass_mark: float = 10, high_absences: int = 10,
                             risk_high: float = 0.60) -> list[Recommendation]:
    recs: list[Recommendation] = []
    g1, g2 = inputs.get("G1"), inputs.get("G2")

    if g2 is not None and g2 < pass_mark:
        recs.append(Recommendation(
            "R1_LOW_LATEST_GRADE", "Focus your revision on core content from the last period",
            "Your latest grade is below the pass mark. Go through the last period's material with your "
            "teacher, list the units you find hardest, and revise those first with past exercises.",
            f"Period 2 grade = {g2:g} (pass mark = {pass_mark:g}).", "high"))
    elif g2 is not None and g2 < pass_mark + 2:
        recs.append(Recommendation(
            "R2_NEAR_PASS_MARK", "You are close to the pass mark - consolidate",
            "A few extra practice sets each week can create a safer margin above the pass mark.",
            f"Period 2 grade = {g2:g}, within 2 points of the pass mark ({pass_mark:g}).", "medium"))

    if g1 is not None and g2 is not None:
        if g2 < g1:
            recs.append(Recommendation(
                "R3_GRADE_DROP", "Find out what changed between period 1 and period 2",
                "Your grade went down. Review the work from period 2, look for recurring mistakes, and "
                "ask for feedback on one or two returned assignments.",
                f"Grade moved from {g1:g} to {g2:g} ({g2 - g1:+g}).", "medium"))
        elif g2 > g1:
            recs.append(Recommendation(
                "R4_IMPROVING", "Keep your momentum",
                "Your grades are improving. Keep your routine and gradually try harder exercises.",
                f"Grade moved from {g1:g} to {g2:g} ({g2 - g1:+g}).", "low"))

    failures = inputs.get("failures")
    if failures is not None and failures >= 1:
        recs.append(Recommendation(
            "R5_PAST_FAILURES", "Rebuild the foundations first",
            "Earlier setbacks can leave gaps that make new topics harder. Spend short, regular sessions "
            "on prerequisite material before moving on.",
            f"Past class failures = {failures:g}.", "medium"))

    absences = inputs.get("absences")
    if absences is not None and absences >= high_absences:
        recs.append(Recommendation(
            "R6_ABSENCES", "Catch up on the lessons you missed",
            "Collect notes and exercises from missed classes (from classmates or your teacher) and "
            "work through them in order.",
            f"Absences = {absences:g} (threshold {high_absences}).", "medium"))

    studytime = inputs.get("studytime")
    if studytime is not None and studytime <= 1:
        recs.append(Recommendation(
            "R7_LOW_STUDY_TIME", "Add a few short study sessions each week",
            "Weekly study time is under 2 hours. Try 2-3 fixed sessions of 30-45 minutes and mix "
            "reading with self-testing.",
            "Weekly study time band = 1 (under 2 hours).", "medium"))

    if inputs.get("goout") is not None and inputs["goout"] >= 4 and (studytime or 0) <= 2:
        recs.append(Recommendation(
            "R8_SCHEDULE_BLOCKS", "Plan study blocks around your social time",
            "Pick fixed times for studying in your weekly plan so that both study and friends fit.",
            f"Going out = {inputs['goout']:g}/5 with study time band = {studytime:g}.", "low"))

    at_risk_context = (probability is not None and probability >= risk_high) or \
                      (g2 is not None and g2 < pass_mark)
    if at_risk_context and inputs.get("schoolsup") == "no":
        recs.append(Recommendation(
            "R9_ASK_SUPPORT", "Ask about extra school support",
            "Many schools offer tutoring or help sessions. Ask your teacher or counsellor what is available.",
            "Extra school support = no, and risk indicators are elevated.", "medium"))
    if at_risk_context and inputs.get("paid") == "no":
        recs.append(Recommendation(
            "R10_EXTRA_CLASSES", "Consider extra practice sessions in this subject",
            "If affordable and available, extra classes or study groups can add structured practice. "
            "Free options such as peer study groups also work.",
            "No extra paid classes in this subject, and risk indicators are elevated.", "low"))

    if probability is not None and probability >= risk_high:
        recs.append(Recommendation(
            "R11_HIGH_MODEL_RISK", "Plan your revision early and talk to your teacher",
            "The model sees a pattern similar to students who ended below the pass mark. This is not a "
            "certainty - use it as a reason to start early and seek feedback.",
            f"Model probability of being at risk = {probability:.0%}.", "high"))

    if not recs:
        recs.append(Recommendation(
            "R0_KEEP_GOING", "Keep up your current routine",
            "No warning signs were found in the available indicators. Regular self-testing keeps knowledge fresh.",
            "No rule thresholds were triggered.", "low"))

    recs.sort(key=lambda r: PRIORITY_ORDER[r.priority])
    return recs
