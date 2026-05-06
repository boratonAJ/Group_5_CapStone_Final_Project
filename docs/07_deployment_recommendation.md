# Deployment Recommendation
## HMDA 2024 Action-Taken Classifier: Q1–Q5 Defensibility Audit

**Version:** 1.0 (Q1–Q5 Complete)
**Date:** May 5, 2026
**Prepared by:** DNSC 6330 Capstone Team
**Course:** DNSC 6330 — Responsible Machine Learning, GWU
**Primary Evidence:** [`notebooks/03_model_audit.ipynb`](../notebooks/03_model_audit.ipynb) (33 cells, all executed)

---

## Summary Recommendation

> **DEPLOY WITH CONDITIONS**
>
> We recommend deployment because the GBM model achieves AUC 0.8127 (test) and the identified fairness violations are remediable via threshold adjustment to 0.20, which achieves AIR ≥ 0.80 across all protected groups. Deployment remains contingent on: (1) threshold implementation (0.20 in production), (2) board approval for updated loan loss reserve (+2-5%), (3) monthly AIR monitoring infrastructure, (4) a GitHub audit record as a governance artifact, and (5) six explicitly named model shutdown triggers with designated decision makers.

This recommendation is based on evidence gathered across Q1–Q5 following the Lecture 06 defensibility framework: a documented objective (Q1), measured disparities (Q3), tested mitigations (Q4), named residual risks (Q4), and a governance plan with monitoring triggers (Q5). It is not a statement that the model is perfect. It is a statement that the evidence is sufficient to support responsible deployment **within defined safeguards**, and that the named conditions must be enforced before any production use.

---

## Evidence Scorecard (Q1–Q5 Audit Findings)

| Dimension | Key Finding | Score | Evidence Source |
|---|---|---|---|
| **Q1: Optimization Objective** | F1 @ threshold 0.2575 (F1 = 0.8861); trade-offs: +35% approval gap for minorities | **YELLOW** | [`notebooks/03_model_audit.ipynb#Q1`](../notebooks/03_model_audit.ipynb#Q1) |
| **Performance (GBM vs. LR)** | GBM AUC 0.8127 vs. LR 0.7442; model substantially more predictive than baseline | **GREEN** | [`notebooks/02_modeling.ipynb`](../notebooks/02_modeling.ipynb) |
| **Q3: Fairness — AIR** | AIR range 0.606–1.000; 3 violations (Free Form Text 0.606, Hawaiian 0.787, Indian 0.800) | **RED** | [`notebooks/03_model_audit.ipynb#Q3`](../notebooks/03_model_audit.ipynb#Q3) |
| **Q3: Fairness — Intersectional** | Black × Female 65% approval vs. White × Male 88%; 23% gap; intersectional harm most severe | **RED** | [`notebooks/03_model_audit.ipynb#Q3`](../notebooks/03_model_audit.ipynb#Q3) |
| **Q4: Mitigation Tested** | Threshold 0.20 remediates AIR to ≥0.80; trade-off: FPR +15-25%, approval 62.8%→91.3% | **GREEN** | [`notebooks/03_model_audit.ipynb#Q4`](../notebooks/03_model_audit.ipynb#Q4) |
| **Q4: Residual Risks Named** | 5 specific risks identified: power loss, correlated features, data quality, drift, intersectional | **GREEN** | [`tables/audit_residual_risks.csv`](../tables/audit_residual_risks.csv) |
| **Q5: Governance & Monitoring** | 6 shutdown triggers defined; GitHub record setup; monthly AIR audits; annual retraining | **GREEN** | [`notebooks/03_model_audit.ipynb#Q5`](../notebooks/03_model_audit.ipynb#Q5) |
| **Documentation** | System card, model card, decision log, risk register, deployment recommendation complete | **GREEN** | [`docs/`](.) |

**Score summary:** 4 Green / 1 Yellow / 2 Red (in Fairness dimensions, but remediable via threshold adjustment + monitoring)

**Decision rule applied:** "Mostly Green with Red in Fairness but documented mitigations → **Deploy with conditions**"
- Fairness violations are measurable and remediable (threshold adjustment)
- Mitigations tested and quantified (AIR improved to ≥0.80)
- Residual risks explicitly named and monitored (5 risks, 6 shutdown triggers)
- Conditions documented and enforceable (monitoring infrastructure, GitHub audit record)

---

## Reasoning by Evidence Dimension

### Q1 — Optimization Objective (YELLOW)

We optimize for **F1 score at threshold 0.2575**, achieving F1 = 0.8861 on the test set. This balances lender precision (minimize default losses) with approval volume (business growth). However, this objective has documented trade-offs:

- **Benefits lenders:** GBM outperforms LR baseline (AUC 0.8127 vs. 0.7442); high precision minimizes defaults
- **Harms minorities:** 38.2% approval gap by race; Free Form Text applicants 24.7% vs. White 62.8%
- **Harms low-risk applicants:** ~129K estimated wrongful denials of low-risk (low-default-probability) applicants

This is appropriate transparency: the objective is clearly stated, its beneficiaries are named, and its harms are quantified. Deployment at this threshold is **NOT recommended** due to fairness violations in Q3.

**Evidence:** [`notebooks/03_model_audit.ipynb#Q1`](../notebooks/03_model_audit.ipynb#Q1), [`tables/metrics_table_final.csv`](../tables/metrics_table_final.csv)

### Q3 — Fairness: AIR Violations (RED → Mitigated to GREEN via Q4)

**Baseline findings (threshold 0.50):**
- AIR violations: 3 groups below 0.80 legal threshold
  - Free Form Text: AIR = **0.606** (39% gap; ~6.5K applicants denied disproportionately)
  - Native Hawaiian: AIR = **0.787** (21% gap; ~200 applicants)
  - American Indian: AIR = **0.800** (20% gap; borderline, ~150 applicants)
- Additional disparities (>20% gap): Black or African American (AIR 0.619), Asian (AIR 0.858)

**Intersectional findings:**
- Black women: 65% approval vs. White men 88% → **23 percentage point gap**
- Black women × High-risk: 40% approval; highest severity subgroup
- Most-harmed intersectional group: 2+ minority races + sex not available = **17.5% approval**

This evidence is a clear fairness violation under ECOA (Equal Credit Opportunity Act) and HMDA fair-lending standards. However, Q4 demonstrates mitigation is effective.

**Evidence:** [`tables/audit_air_by_race.csv`](../tables/audit_air_by_race.csv), [`figures/audit_air_disparate_impact.png`](../figures/audit_air_disparate_impact.png)

### Q4 — Mitigation #1: Threshold Adjustment (GREEN)

**Strategy:** We raise the prediction threshold from 0.50 → **0.20** to achieve demographic parity (equalize approval rates).

**Results:**
- Free Form Text: AIR 0.606 → **0.82** (passes 80% rule)
- Native Hawaiian: AIR 0.787 → **0.85**
- American Indian: AIR 0.800 → **0.82**
- All protected groups: AIR ≥ 0.80

**Trade-off (acceptable with conditions):**
- Overall approval rate: 62.8% → 91.3% (lenders approve more borderline applicants)
- Lender false positive rate (FPR) among approved: increases from baseline to **19.1%** (+15-25% estimated)
- Model predictive power: AUC remains ~0.81; F1 remains ~0.89 (acceptable trade-off)

**Justification:** Raising the threshold favors approval, which disproportionately helps protected groups in the borderline region (where they are overrepresented in rejection). This achieves the ECOA objective: equal treatment under law.

**Evidence:** [`tables/audit_mitigation_threshold_adjustment.csv`](../tables/audit_mitigation_threshold_adjustment.csv), [`figures/audit_before_after_mitigation.png`](../figures/audit_before_after_mitigation.png)

### Q4 — Residual Risks (5 Named Risks = GREEN)

Even after mitigation, 5 specific residual risks remain. They are accepted **with monitoring conditions**:

1. **Reduced Predictive Power:** Threshold adjustment increases FPR by 15-25%; lenders absorb more defaults
   - Monitoring: Monthly loss rate tracking by score decile
   - Acceptance condition: Loss rate ≤ X% (board-approved threshold)

2. **Remaining Correlated Features:** Income, DTI, loan-to-value may still correlate with race indirectly
   - Monitoring: Quarterly feature correlation audit (flag |r| > 0.15 to protected attributes)
   - Mitigation path: Feature removal considered for future retraining (deferred)

3. **Data Quality Issues:** Income, employment self-reported; quality varies by demographic (gig workers, informal employment)
   - Monitoring: Track data completeness by demographic group; flag missing rate >20%
   - Acceptance condition: Enhanced data collection for underrepresented segments

4. **Future Demographic Drift:** Population demographics shift; model trained on 2024 data only
   - Magnitude: Estimated 1-2% AUC loss per year; AIR violations likely within 2-3 years if not retrained
   - Monitoring: Annual demographic drift test (Wasserstein distance > 0.05 = trigger retraining)

5. **Intersectional Disparities:** Race × sex combinations may remain underserved even after threshold adjustment
   - Monitoring: Quarterly intersectional audit (race × sex × ethnicity breakdown)
   - Acceptance condition: AIR ≥ 0.80 maintained across **all intersectional combinations** in monitoring data

**Evidence:** [`tables/audit_residual_risks.csv`](../tables/audit_residual_risks.csv), [`tables/audit_acceptance_conditions.txt`](../tables/audit_acceptance_conditions.txt)

### Q5 — Governance & Monitoring (GREEN)

**Deployment requires:**

1. **Threshold Implementation:** Deploy 0.20 in production; document ECOA business justification
2. **Risk Appetite Board Approval:** Authorize +2-5% loan loss reserve increase
3. **Monitoring Infrastructure:** Real-time AIR dashboard; monthly audits; automated alerts if AIR < 0.75
4. **GitHub Audit Record:** Repository as canonical governance artifact; branch protection; monthly monitoring logs committed
5. **Stakeholder Alignment:** Legal/compliance sign-off; loan officer training; risk committee briefing

**Shutdown Triggers (6 escalation paths):**
| Trigger | Threshold | Response | Decision Maker |
|---------|-----------|----------|---|
| SHUT-1: Regulatory | CFPB/DOJ CID or investigation notice | Halt scoring within 48h | CRO + Legal |
| SHUT-2: AIR Collapse | AIR < 0.75 for 2+ consecutive months | 30-day urgent review; potential pause | Compliance |
| SHUT-3: Loss Breach | Loss rate >50 bps above risk appetite | Quarterly review; retrain or adjust threshold | Risk Committee |
| SHUT-4: Demographic Drift | Wasserstein distance > 0.05 (annual) | Mandatory retraining; validate fairness pre-deploy | Data Science |
| SHUT-5: Public/Media | Civil rights complaint or news investigation | Emergency audit within 48h; external review | CRO + Communications |
| SHUT-6: AUC Degradation | AUC drops below 0.75 on holdout | Investigate within 2 weeks; retrain or deprecate | Data Science |

**Evidence:** [`tables/audit_shutdown_triggers.csv`](../tables/audit_shutdown_triggers.csv), [`tables/audit_deployment_checklist.csv`](../tables/audit_deployment_checklist.csv)

---

## Conditions for Deployment (ALL Required)

**CONDITION 1: Threshold Implementation & Documentation**
- Deploy P(default) < 0.20 in production scoring engine (not 0.50)
- Update decision rules in loan origination system
- Document business justification: "Threshold optimized for ECOA compliance — equal opportunity lender standard"
- Compliance and legal sign-off required before first scoring batch

**CONDITION 2: Risk Appetite Board Approval**
- Board resolution approving updated risk appetite: Expected loan loss rate increase from [baseline] → [new]%
- Loan loss reserve increase: +2–5% of annual origination volume
- Finance team staffed for monthly loss tracking by demographic group
- Contingency plan if actual losses exceed reserve

**CONDITION 3: Monitoring Infrastructure (Pre-Deployment)**
- Real-time AIR tracking dashboard deployed and validated
- Monthly AIR computation by race, sex, ethnicity, and intersections
- Automated alerts: IF AIR < 0.75 for any protected group, escalate to Chief Compliance Officer
- Quarterly demographic drift test (Wasserstein distance computation)
- Annual AUC recomputation on held-out test data

**CONDITION 4: GitHub Audit Record Setup**
- Repository: `boratonAJ/Group_5_CapStone_Final_Project` on GitHub
- Branch protection: `main` requires ≥2 approvals before merge
- Standardized commit tags: `audit-v1.0-2026-05-05` (version + date)
- Monthly monitoring logs committed: `monitoring/monthly_air.csv` updated each month
- README linked to all fairness metrics, deployment conditions, monitoring schedule
- Annual retraining reports filed in `docs/`
- **This repository is the authoritative audit record for examiner requests (CFPB, DOJ, state AG)**

**CONDITION 5: Stakeholder Communication & Training**
- Compliance team: Briefed on AIR metric, monitoring triggers, pause conditions
- Risk committee: Updated on loan loss estimates, demographic drift monitoring
- Loan officers: Trained on consistent threshold scoring; no override permissions without escalation
- Legal/compliance: Sign-off letter on file; deployment date locked in before go-live

**CONDITION 6: Examiner Readiness**
- Model card updated with Q1–Q5 findings: objectives, fairness metrics, mitigations, residual risks
- CFPB/DOJ audit response prepared: Links to GitHub repo, monitoring logs, fairness documentation
- Fair lending audit schedule: Quarterly internal audits + annual external fairness validation

---

## What Would Trigger Model Pause or Redesign

This model **must be paused** if any of the following occur:

| Trigger | Threshold | Action | Timeline |
|---------|-----------|--------|----------|
| **AIR Violation** | Any protected group AIR < 0.75 for 2+ consecutive monthly periods | IMMEDIATE PAUSE; fairness re-audit | 30 days |
| **Regulatory Investigation** | CFPB/DOJ/State AG CID or formal notice | IMMEDIATE SHUTDOWN; halt scoring within 48h | Same day |
| **Loss Rate Breach** | Observed loss rate exceeds risk appetite by >50 basis points | Escalate to risk committee; review & reestimate | Quarterly |
| **Demographic Drift** | Wasserstein distance > 0.05 (annual test) | Trigger mandatory retraining; validate fairness | Annual (Q4) |
| **Public Complaint/Media** | Civil rights organization or press identifies disparity | Emergency audit within 48h; external review | 30 days |
| **AUC Degradation** | AUC drops below 0.75 on quarterly holdout evaluation | Investigate root cause; retrain or deprecate | 2 weeks |

---

## Residual Risks Accepted Under This Recommendation

The following risks are accepted. They define the boundary conditions of deployment and are documented in the risk register.

| Risk | Magnitude | Mitigation | Monitoring Frequency |
|------|-----------|-----------|---|
| Reduced predictive power (FPR) | +15-25% false positives | Loss reserve increased 2-5% | Monthly |
| Correlated features (income, DTI) | 20-40% of bias may remain | Feature correlation audits | Quarterly |
| Data quality (self-reported income) | 10-20% unexplained disparities | Track completeness by demographic | Quarterly |
| Demographic drift | 1-2% AUC loss per year | Annual demographic shift test | Annual |
| Intersectional disparities | May persist for race × sex combinations | Intersectional audit | Quarterly |

---

## Summary: DEPLOY WITH CONDITIONS

**This model is defensible for deployment because:**

1. Objective is transparent (Q1): F1 optimization with acknowledged trade-offs
2. Disparities are measured and quantified (Q3): AIR violations explicitly identified
3. Mitigations are tested and effective (Q4): Threshold adjustment remediates violations; AIR ≥ 0.80
4. Residual risks are named and monitored (Q4–Q5): 5 risks with monitoring conditions; 6 shutdown triggers
5. Governance is explicit (Q5): GitHub audit record, monthly monitoring, annual retraining, decision-maker roles

**Deployment is NOT defensible if any condition is unmet.** Failure to implement monitoring, GitHub record, or threshold in production = failure to uphold fair-lending standard.

**Evidence for examiner response:** Link all GitHub monitoring logs, monthly AIR reports, and retraining decisions to demonstrate ongoing compliance.

---

**Recommendation prepared:** May 5, 2026  
**Model version:** GBM v20260505  
**Threshold:** 0.20 (demographic parity optimized)  
**Status:** **READY FOR CONDITIONAL DEPLOYMENT**

---

## Conditions for Deployment

**All of the following conditions must be satisfied before any production use:**

1. **Monthly AIR monitoring** must be operational before the first scoring batch.
   - Platform: [specify monitoring system]
   - Pause trigger: AIR < 0.80 for any protected group in two consecutive monthly periods.

2. **Protected attributes must be captured at scoring time** and stored for retrospective monitoring.
   - If this is not feasible, proxy-free monitoring (PSI, score distribution) must substitute, and the monitoring plan must be updated to reflect reduced fairness observability.

3. **Quarterly PSI monitoring** must be operational for `loan_amount` and `income`.
   - Review trigger: PSI > 0.25 on either feature.

4. **Proxy-risk feature monitoring:** If `tract_minority_population_percent` or any High-proxy-risk feature rises into the top 5 SHAP features in any monitoring period, a fairness re-audit is required before continued deployment.

5. **Model card and deployment recommendation** must be reviewed and re-signed:
   - After 12 months of deployment, OR
   - After any material change in lending policy, feature engineering, or training data, OR
   - After any regulatory guidance change from CFPB, HUD, or OCC that affects HMDA fair-lending analysis.

6. **Broker-channel anomaly detection** must be implemented before high-volume deployment.
   - Applies specifically to threat scenario T-01 (income/DTI gaming).

---

## Pause and Reverse Conditions

**This model must be paused or taken offline if any of the following are observed:**

| Trigger | Threshold | Required Action |
|---|---|---|
| AIR for any protected group | < 0.80 for 2 consecutive monthly monitoring periods | IMMEDIATE PAUSE; fairness re-audit within 30 days |
| PSI for loan_amount or income | > 0.25 in any quarterly review | Pause retraining; feature review before any model update |
| FNR gap vs. reference group widens | > 20% relative change for any minority group | Escalate to compliance; potential pause pending investigation |
| New top-5 SHAP feature with proxy risk ≥ Medium | Correlation |r| > 0.30 with any protected attribute | Fairness re-audit before next deployment cycle |
| Training data integrity compromise | Any confirmed incident | IMMEDIATE PAUSE; full audit of affected scoring period |
| Regulatory guidance change | CFPB/HUD issues new fair-lending guidance affecting HMDA models | Compliance review within 60 days; potential retrain |
| Model AUC | Drops below 0.68 in any monthly evaluation | Retrain review initiated within 30 days |

---

## Residual Risks Accepted Under This Recommendation

The following risks are known and accepted. They are the documented boundary conditions of this deployment decision.

| Risk ID | Risk | Why Accepted |
|---|---|---|
| RR-002 | Small intersectional cells (AIAN, NHPI) | Acknowledged; monitoring will accumulate data over time |
| RR-003 | Single training year | Geographic holdout tested; annual retrain planned |
| RR-010 | DTI bucketing precision loss | Fundamental HMDA data limitation; no alternative available |

See `docs/residual_risk_register.md` for the full register.

---

## What Would Change This Recommendation

The following findings, if discovered, would move the recommendation to **Do Not Deploy Yet** or **Pause and Redesign**:

- AIR for Black or Hispanic applicants below 0.70 at the operating threshold.
- FNR gap exceeding 30% between any minority group and the White reference group.
- Evidence that a High-proxy feature is necessary for acceptable performance (i.e., removing it causes AUC to fall below 0.68).
- PSI > 0.25 on the geographic holdout for any core creditworthiness feature before any mitigation.
- Any unmitigated High-severity security vulnerability affecting scoring integrity.

---

## Signatures (for institutional deployment)

| Role | Name | Date | Signature |
|---|---|---|---|
| Lead Analyst | | | |
| Fairness Officer | | | |
| Model Risk Officer | | | |
| Compliance Review | | | |
