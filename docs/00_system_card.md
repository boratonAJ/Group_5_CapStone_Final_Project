# System Card: HMDA 2024 Action-Taken Classifier

**Version:** 1.0  
**Date:** April 2026  
**Course:** DNSC 6330 — Responsible Machine Learning, George Washington University  
**Framework:** Lecture 06 — Measurement before opinion · Diagnostics before remediation · Documentation before deployment

---

## 1. System Description

This system is a binary classification model trained on the 2024 HMDA Loan/Application Records (LAR).  
It predicts the likelihood that a mortgage application results in **loan origination or approval**  
(label = 1) versus a **formal denial** (label = 0).

The model is intended to support mortgage credit-decision review workflows in a regulated fair-lending context. It is **not** a substitute for human underwriting judgment and is **not** designed to replace legally required adverse action notices.

---

## 2. Label Definition

| Raw `action_taken` | Meaning | Model Label | Retained |
|---|---|---|---|
| 1 | Loan originated | 1 | ✓ |
| 2 | Application approved, not accepted | 1 | ✓ |
| 3 | Application denied by financial institution | 0 | ✓ |
| 4 | Application withdrawn by applicant | — | Filtered out |
| 5 | File closed for incompleteness | — | Filtered out |
| 6 | Purchased loan | — | Filtered out |
| 7 | Preapproval request denied | — | Filtered out |
| 8 | Preapproval request approved but not accepted | — | Filtered out |

**Rationale for filtering:** Values 4–8 represent outcomes where the lender's decision cannot be cleanly characterized as approval or denial. Including them would introduce label ambiguity and conflate voluntary withdrawal with lender denial — a critical distinction in fair-lending analysis.

**Label note:** Values 1 and 2 are combined as the positive class (y=1) because both represent a positive credit decision by the lender. The distinction between origination and approval-not-accepted reflects applicant behavior after the credit decision, not the lender's creditworthiness assessment.

---

## 3. Optimization Objective

The model is optimized to maximize AUC-ROC on the validation set subject to the constraint that Adverse Impact Ratios remain ≥ 0.80 across all protected groups defined by `derived_race`, `derived_sex`, and `derived_ethnicity`.

**A model that achieves higher AUC at the cost of AIR < 0.80 for any material protected group is not considered acceptable for deployment under this system card.**

This reflects the Lecture 01 principle that the optimization objective must incorporate competing interests, not only accuracy.

---

## 4. Failure Definition

| Error Type | Direction | Primary Harm | Severity | Who Bears It |
|---|---|---|---|---|
| False Negative | Model predicts denial; loan would have been approved | Qualified applicant denied credit | High | Applicant |
| False Positive | Model predicts approval; loan would have been denied | Under-qualified applicant proceeds; institution risk | Medium | Institution |
| Proxy reliance | Model relies on geographic or income features to encode race | Facially neutral disparate impact | High | Minority applicants |
| Distributional drift | Model degrades silently as population shifts | Undetected fairness and performance degradation | High | Minority applicants + institution |
| Calibration gap | Model probabilities are biased for a subgroup | Miscalibrated risk scores affect downstream decisions | Medium | Subgroup |

**Asymmetric harm note:** False negatives are more severe than false positives in the fair-lending context, particularly when FNR is elevated for racial or ethnic minority groups. This asymmetry should inform threshold selection.

---

## 5. Stakeholder Map

| Stakeholder | Interest | Risk if Model Fails |
|---|---|---|
| Mortgage applicants (all) | Fair, accurate evaluation of creditworthiness | Wrongful denial; wasted application costs |
| Applicants from protected groups | Equal treatment regardless of race, sex, ethnicity | Disparate impact; ECOA/FHA violation |
| Lending institution | Accurate credit risk prediction; compliance | Regulatory enforcement; reputational harm |
| Model Risk Officers (MRO) | Model is validated, monitored, documented | Undetected degradation; audit failure |
| Regulators (CFPB, HUD, OCC) | ECOA and FHA compliance | Undetected fair-lending violations |
| Mortgage brokers | Efficient loan processing | N/A (also potential gaming threat actor) |
| Third-party data vendors | Contract revenue | Data integrity risks; poisoning vector |

---

## 6. Data Scope

| Item | Value |
|---|---|
| Dataset | 2024 HMDA Loan/Application Records (LAR) |
| Source | CFPB HMDA Data Browser — https://ffiec.cfpb.gov/data-browser/ |
| Format | Pipe-delimited text (2024_lar.txt), 99 columns |
| Population | All US mortgage applications submitted to HMDA-reporting institutions in 2024 |
| Protected attributes | `derived_race`, `derived_sex`, `derived_ethnicity` (lender-reported derived fields) |

**Protected attribute limitation:** HMDA protected attributes are derived by lenders from self-reported applicant data, government monitoring forms, and visual observation. They are not directly self-reported by applicants in all cases. This introduces measurement error for applicants coded as "Race Not Available," "Sex Not Available," or "Ethnicity Not Available."

**Exact `derived_race` values in 2024 LAR:** White, Black or African American, Asian, American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, 2 or more minority races, Joint, Free Form Text Only, Race Not Available.  
**Exact `derived_sex` values:** Male, Female, Joint, Sex Not Available.  
**Exact `derived_ethnicity` values:** Not Hispanic or Latino, Hispanic or Latino, Joint, Free Form Text Only, Ethnicity Not Available.

---

## 7. Out-of-Scope Uses

This model is **NOT** designed or validated for:
- Pricing or interest-rate setting
- Post-origination default prediction
- Pre-screening or pre-qualification before a formal HMDA-reportable application
- Deployment on non-2024 HMDA data without revalidation
- Automated denial without human review
- Any jurisdiction-specific regulatory compliance determination without legal review

---

## 8. Known Limitations at Scoping

1. **Single-year snapshot:** Training on 2024 LAR only; no temporal validation. Geographic holdout used as proxy for distribution shift.
2. **Geography as proxy:** Census tract, MSA, and state features may encode residential segregation patterns that correlate with race.
3. **DTI bucketing:** `debt_to_income_ratio` is reported as string buckets in HMDA, losing precision at the individual level.
4. **Derived attributes:** Protected attributes derived from lender-reported fields introduce measurement error for applicants coded as "Race Not Available," "Sex Not Available," or "Ethnicity Not Available" — these are retained as their own analytical group, not dropped.
5. **Intersectional sample sizes:** Some intersectional subgroups (e.g., AIAN × Female) have small samples; estimates are directional only.
6. **Co-applicant:** Analysis focuses on primary applicant attributes. Co-applicant demographics are tracked but not primary in the fairness audit.

---

## 9. Decision Log Reference

All design decisions (label construction, leakage removal, split strategy, threshold selection) are recorded in `docs/decision_log.md`. See that document for the reasoning behind each material choice.
