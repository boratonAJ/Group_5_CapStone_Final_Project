# Residual Risk Register — HMDA 2024 Responsible ML Capstone

**Purpose:** We document all known, open, and accepted residual risks at the time of the deployment recommendation.
A risk on this register is not evidence of model failure — it is evidence that risk is known, documented, and owned.

**Severity scale:** High / Medium / Low  
**Likelihood scale:** High / Medium / Low / Certain  
**Status:** Open / Mitigated / Accepted / Escalated

---

| ID | Risk Description | Category | Severity | Likelihood | Evidence | Mitigation | Residual After Mitigation | Owner | Status |
|---|---|---|---|---|---|---|---|---|---|
| RR-001 | Geographic/census-tract features may proxy for race (derived_race correlation) | Proxy Risk / Fairness | High | Medium | SHAP rank + Pearson correlation in Notebook 03 | Monitor SHAP rank; flag if census_tract enters top 5 | Medium — monitoring does not eliminate proxy encoding | Fairness Officer | Open — resolved in Phase 3 |
| RR-002 | Intersectional subgroups (AIAN, NHPI) have small sample sizes; AIR estimates unreliable | Fairness / Data Quality | Medium | Certain | Sample size table in Notebook 04 | Report with n-suppression caveat (n < 30); flag for post-deployment monitoring as data accumulates | Low — acknowledged and documented | Fairness Officer | Accepted |
| RR-003 | Single training year (2024 LAR) — no temporal validation possible | Robustness | Medium | Certain | By design; geographic holdout used as proxy | Annual retrain with new LAR data; geographic holdout PSI testing | Medium — geographic holdout is imperfect temporal proxy | ML Engineer | Accepted |
| RR-004 | Broker gaming: brokers may learn to inflate income/DTI to flip borderline decisions | Security / Input Integrity | Medium | High | Threat scenario T-01 in Notebook 06 | Plausibility checks on income/DTI; outlier flagging on high-volume broker channels | Medium — plausibility checks are imperfect | Compliance | Open |
| RR-005 | Calibration gap between racial subgroups may persist post-deployment | Fairness / Calibration | Medium | Medium | Subgroup ECE analysis in Notebook 05 | Quarterly calibration monitoring with ECE review trigger | Low — if monitoring trigger functions as designed | Fairness Officer | Open |
| RR-006 | Third-party vendor data (census tract, ACS variables) could be corrupted during retraining | Security / Data Integrity | Medium | Low | Threat scenario T-02 in Notebook 06 | Hash validation of external data files; PSI check before retraining | Low | Data Owner | Open |
| RR-007 | Protected attributes may not be available at scoring time in all deployment contexts | Monitoring Feasibility | High | Medium | Assumption documented in monitoring playbook | Require protected-attr capture at scoring time as deployment condition | Medium — if condition is not enforced, fairness monitoring breaks | Compliance | Conditional |
| RR-008 | Operating threshold fixed at [THRESHOLD] — may need adjustment if approval-rate benchmarks change | Performance / Fairness | Low | Low | Decision D-007 in decision log | Annual threshold review as part of model validation cycle | Low | ML Engineer | Accepted |
| RR-009 | "Race/Sex/Ethnicity Not Available" applicants are treated as their own group; their true protected status is unknown. Also, "Joint" applicants (both applicant and co-applicant demographics combined) cannot be attributed to a single protected group. | Fairness / Data Quality | Medium | Certain | HMDA data limitation documented in system card | We report these as separate groups ("Race Not Available", "Sex Not Available", "Ethnicity Not Available", "Joint"); disparity analysis is flagged as incomplete for these segments | Medium — fundamental HMDA data limitation | Fairness Officer | Accepted |
| RR-010 | DTI reported as string buckets — loses precision at individual level | Model Quality | Low | Certain | Feature engineering note in Notebook 01 | Numeric midpoint imputation; flagged in model card | Low | Modeler | Accepted |

---

## Summary at Time of Recommendation

| Category | Open | Mitigated | Accepted |
|---|---|---|---|
| Proxy Risk / Fairness | 1 | 0 | 1 |
| Robustness | 0 | 0 | 1 |
| Security | 2 | 0 | 0 |
| Monitoring Feasibility | 0 | 0 | 1 |
| Data Quality | 0 | 0 | 3 |

**Unmitigated High-severity risks:** [To be filled after Phase 3–6 analysis]  
**Deployment-blocking risks:** Any residual risk rated High severity + High likelihood with no mitigation.
