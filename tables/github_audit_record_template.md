# HMDA Mortgage Model: Responsible AI Audit Record

**Model Name:** GBM HMDA Mortgage Approval Prediction v1.0
**Deployment Date:** [TBD]
**Threshold:** 0.20 (optimized for demographic parity)
**Decision:** DEPLOY WITH CONDITIONS

## Audit Artifacts

### Q1 — Optimization Objective
- Metric: F1 score @ threshold 0.2575 (test set F1 = 0.8861)
- Business Rationale: Balance lender precision (minimize defaults) with approval volume
- Trade-offs: Benefits lenders (high precision) vs. harms minorities & low-risk applicants
- File: `tables/metrics_table_final.csv`

### Q3 — Subgroup Error Measurement
- AIR violations detected: Free Form Text Only (0.606), Native Hawaiian (0.787), American Indian (0.800)
- Fairness metrics: AIR, ME, SMD by race, sex, ethnicity
- Intersectional disparities: Black women 65% approval vs. White men 88%
- Files: `tables/audit_air_*.csv`, `figures/audit_air_disparate_impact.png`

### Q4 — Residual Risks After Mitigation
- Mitigation #1: Threshold adjustment to 0.20 achieves AIR ≥ 0.80
- Mitigation #2: Feature removal (deferred to next iteration; requires retraining)
- Residual risks: Reduced predictive power, remaining correlated features, demographic drift
- Files: `tables/audit_residual_risks.csv`, `tables/audit_acceptance_conditions.txt`

### Q5 — Deployment Defensibility
- Recommendation: DEPLOY WITH CONDITIONS
- Mandatory conditions: Threshold (0.20), risk appetite update, monitoring, GitHub record, stakeholder training
- Shutdown triggers: Regulatory investigation, AIR collapse, loan loss breach, demographic drift, public complaint, AUC degradation
- Files: `tables/audit_deployment_checklist.csv`, `tables/audit_shutdown_triggers.csv`

## Governance Artifacts

### Deployment Sign-Off
```
Threshold: 0.20 (effective [DATE])
Risk Appetite: Approved by Board resolution [DATE]
Legal Review: Approved by General Counsel [DATE]
Monitoring: Dashboard live by [DATE]
```

### Monthly Monitoring (Automated)
- Updated each month in `monitoring/monthly_air.csv`
- Alert if AIR < 0.75 for any protected group

### Annual Review
- Demographic drift test (Wasserstein distance)
- Model retraining decision
- Fairness validation

## Contact

**Model Owner:** [Data Science Lead]
**Compliance Owner:** [Compliance Officer]
**Emergency Contact:** [Chief Risk Officer]

---
Last Updated: [DATE]
Version: 1.0
Repository: responsible-lending/hmda-capstone-audit
