import unittest

from src.recommendations import generate_recommendations


def ids(recs):
    return [r.rule_id for r in recs]


BASE = dict(G1=12, G2=13, studytime=3, absences=2, failures=0, goout=2, schoolsup="yes", paid="yes")


class RecommendationTests(unittest.TestCase):
    def test_low_grade_triggers_high_priority(self):
        recs = generate_recommendations({**BASE, "G1": 9, "G2": 7}, 0.8)
        self.assertIn("R1_LOW_LATEST_GRADE", ids(recs))
        self.assertEqual(recs[0].priority, "high")

    def test_grade_drop_and_improvement(self):
        self.assertIn("R3_GRADE_DROP", ids(generate_recommendations({**BASE, "G1": 15, "G2": 12}, 0.1)))
        self.assertIn("R4_IMPROVING", ids(generate_recommendations({**BASE, "G1": 11, "G2": 14}, 0.1)))

    def test_absences_failures_and_study_time(self):
        recs = ids(generate_recommendations({**BASE, "absences": 15, "failures": 2, "studytime": 1}, 0.3))
        for rule in ("R6_ABSENCES", "R5_PAST_FAILURES", "R7_LOW_STUDY_TIME"):
            self.assertIn(rule, recs)

    def test_support_rules_only_when_risk_is_elevated(self):
        inputs = {**BASE, "schoolsup": "no", "paid": "no"}
        self.assertNotIn("R9_ASK_SUPPORT", ids(generate_recommendations(inputs, 0.1)))
        self.assertIn("R9_ASK_SUPPORT", ids(generate_recommendations(inputs, 0.9)))

    def test_no_warning_signs_gives_default_message(self):
        recs = generate_recommendations({**BASE, "G1": 14, "G2": 14}, 0.05)
        self.assertEqual(ids(recs), ["R0_KEEP_GOING"])

    def test_works_without_prior_grades_and_probability(self):
        recs = generate_recommendations({"studytime": 1, "absences": 12, "failures": 0}, None)
        self.assertNotIn("R1_LOW_LATEST_GRADE", ids(recs))
        self.assertIn("R7_LOW_STUDY_TIME", ids(recs))

    def test_every_recommendation_has_evidence_and_is_ordered(self):
        recs = generate_recommendations({**BASE, "G1": 9, "G2": 6, "absences": 20, "studytime": 1}, 0.9)
        order = {"high": 0, "medium": 1, "low": 2}
        self.assertEqual([order[r.priority] for r in recs], sorted(order[r.priority] for r in recs))
        for r in recs:
            self.assertTrue(r.evidence and r.advice and r.title)

    def test_language_is_not_judgmental(self):
        text = " ".join(r.advice + r.title for r in generate_recommendations({**BASE, "G2": 4}, 0.9)).lower()
        for word in ("lazy", "stupid", "unintelligent", "incapable"):
            self.assertNotIn(word, text)


if __name__ == "__main__":
    unittest.main()
