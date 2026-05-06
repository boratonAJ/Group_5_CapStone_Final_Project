# Security Assessment and Residual Risk Analysis
## HMDA 2024 Action-Taken Classifier

**Version:** 1.0  
**Prepared by:** [TEAM NAMES]  
**Date:** [DATE]  
**Course:** DNSC 6330 — Responsible Machine Learning, GWU

---

## 1. Threat Actor Framework

We treat a deployed HMDA scoring model as an adversarial system. Unlike a generic ML system, an HMDA-linked model sits at the intersection of financial incentives, consumer rights, and regulatory scrutiny. The following categories of threat actors are relevant:

| Threat Actor | Motivation | Capability | Entry Point |
|---|---|---|---|
| Mortgage brokers | Maximize client approval rates for commission income | High domain knowledge; repeated system access | Application fields before submission |
| Loan officers | Approval quota pressure; avoid underwriting liability | Operational access to system inputs | Manual input fields; system overrides |
| Sophisticated applicants | Self-optimize application to maximize approval probability | Low–Medium; depends on publicly available information | Public HMDA data; observable decision patterns |
| Third-party data vendors | Contract revenue; indifference to data quality | Access to training data inputs (census, ACS features) | Training data pipeline |
| Internal adversaries | Cover discriminatory practices; manipulate model to suppress minority approvals | High — insider access | Training data; model configuration; label pipeline |
| External attackers | Model extraction; competitive intelligence; intellectual property | Medium; requires repeated API access | Scoring API / portal |

---

## 2. Threat Scenario Table

| ID | Scenario | Actor | Entry Point | Impact | Likelihood | Mitigation Required | Residual Risk |
|---|---|---|---|---|---|---|---|
| T-01 | **Income / DTI inflation by broker:** Broker coaches applicant to inflate stated income or understate DTI fields to flip borderline credit decisions | Mortgage broker | `income`, `debt_to_income_ratio` fields at application entry | Inflated approvals; increased credit risk; model gaming spreads through broker channel | High | Income plausibility checks; outlier flagging on high-volume broker channels; third-party income verification | Medium |
| T-02 | **Poisoning via vendor data corruption:** Third-party vendor provides corrupted or systematically biased census-tract features during a training data refresh | Third-party data vendor | Training data pipeline (ACS / census features) | Proxy-risk amplification; disparate outcomes shift without triggering performance alerts | Medium | SHA-256 hash validation of all external data files; PSI check against prior training distribution before retraining; mandatory human review if PSI > 0.10 on proxy-risk features | Low–Medium |
| T-03 | **Scoring API extraction:** External actor submits large volumes of synthetic applications to a scoring endpoint to reverse-engineer the model's decision boundary | External attacker | Production scoring API / portal | Model intellectual property exfiltration; enables systematic gaming guide for protected-group boundary | Low–Medium | Rate-limiting on scoring API; anomaly detection on query volume and feature distribution; score-difference suppression in API responses | Low |
| T-04 | **Label manipulation by insider:** Internal employee with access to training data or label pipeline modifies `action_taken` labels to systematically suppress minority approval rates in training data | Internal adversary | Training data or label construction pipeline | Discriminatory model encoded silently; disparate outcomes may be invisible at model-level monitoring | Low | Audit trail on all label modifications; 4-eyes principle for training data changes; automated detection of label distribution shift between training versions | Low |
| T-05 | **Boundary probing by broker consortium:** Multiple brokers coordinate to submit threshold-boundary applications, pooling information to reconstruct the decision surface for a protected group | Broker consortium | Repeated scoring via application portal | Targeted gaming of decision boundary near AIR = 0.80; model behavior for minority applicants specifically mapped | Medium | Score-smoothing (return categorical decision, not raw score); threshold obfuscation; broker-channel monitoring for correlated application patterns | Medium |

---

## 3. Gaming / Input Manipulation Detail (T-01)

We identify field-level gaming by mortgage brokers as the most realistic near-term threat. HMDA-reportable fields that feed the model include:

- `income` (stated by applicant; broker can coach)
- `debt_to_income_ratio` (derived from stated income and debt; inflatable)
- `loan_amount` (partially strategic; broker may adjust to optimize LTV ratio)
- `combined_loan_to_value_ratio` (connected to property_value and loan_amount)

Brokers who learn which fields most strongly influence the model score (via SHAP or published guidance) can coach applicants to optimize their inputs. This produces applications that score well but do not reflect underlying credit risk.

**Compounding fair-lending concern:** If gaming behavior is concentrated in majority-group broker channels but not minority-group channels (due to information asymmetry), gaming could *increase* approval rates in ways that inadvertently improve or worsen AIR depending on the direction.

**Mitigations:**
1. Income and DTI plausibility checks against third-party verification data (IRS transcripts, credit bureau).
2. Anomaly detection: flag applications from brokers whose clients show unusual income-to-DTI ratios.
3. Regular model re-evaluation stratified by broker channel.

---

## 4. Poisoning Resistance

If we retrain the model periodically using new HMDA LAR data, a third-party vendor supplying ACS or census-tract features could introduce systematic bias into geographic variables. Because census-tract features carry elevated proxy risk for race, corrupting those features during retraining could shift the model's disparate impact characteristics without triggering performance-level alerts.

**Why PSI alone is insufficient:** PSI detects distribution shift, but a systematically biased (internally consistent) dataset may have low PSI while encoding a new racial bias pattern.

**Required controls:**
1. SHA-256 file hash validation for all external data files before pipeline execution.
2. Automated PSI check comparing new training data to prior training data before retraining proceeds.
3. Mandatory human review of any PSI > 0.10 on features rated Medium or High proxy risk.
4. Separate pre-retrain fairness check: run AIR on new training data before model is updated.

---

## 5. Access-Control Specification

The following access-control matrix defines minimum required access controls for any deployment of this model. Deviations require documented justification reviewed by the Model Risk Officer.

| Role | Train Model | Score Application | View Raw Features | View Prediction Score | View Protected Attributes | View Model Weights | Modify Training Data |
|---|---|---|---|---|---|---|---|
| Data Engineer | YES (pipeline only) | NO | YES | NO | Masked | NO | YES (with audit trail) |
| ML Engineer | YES | YES (test/staging) | YES | YES | Masked | YES (read) | YES (with 4-eyes) |
| Underwriter | NO | View output only | NO | Binary only (approve/deny) | NO | NO | NO |
| Compliance Officer | NO | NO | Aggregated reports | YES (aggregate) | YES (aggregate) | NO | NO |
| Model Risk Officer | NO (audit only) | YES (validation testing) | YES | YES | YES (for validation) | YES (read) | NO |
| External Auditor | NO | NO | Aggregated only | Aggregated only | YES (aggregate) | NO | NO |
| Applicant | NO | NO | Own record only | Denial reason (legally required) | Own record only | NO | NO |

**Key control principles:**
- Protected attribute data must never be exposed to scoring-role users who could use it in real-time decisions.
- Training data modifications require a 4-eyes principle review and an immutable audit entry.
- Scoring API returns should be binary (approve/deny) or probability-bucketed — not raw probabilities — to limit extraction risk.

---

## 6. Adversarial Failure Modes Summary

The three most plausible adversarial failure modes for a deployed HMDA scoring system are:

1. **Silent gaming normalization:** Broker gaming of income/DTI fields becomes industry-standard practice. The model's effective training distribution shifts over time to include inflated inputs, requiring periodic recalibration. If this gaming is not race-neutral, it may silently affect AIR.

2. **Proxy feature drift:** A geographic rezoning event, census tract boundary change, or neighborhood demographic shift changes the meaning of census-tract features the model relies on. The model's predictions change without any change in the model itself, and PSI may not detect this if the shift is gradual.

3. **Monitoring system failure:** Protected attribute data is not captured at scoring time (common if the scoring system and the application intake system are separate). AIR monitoring becomes impossible. The model continues scoring without fairness oversight.

---

## 7. Residual Risk Summary

See `docs/residual_risk_register.md` for the full register. The following risks are specifically attributable to the security threat landscape:

- **RR-004** (Broker gaming): Open; we recommend mitigations, but they are not implemented in this audit.
- **RR-006** (Vendor data poisoning): Open; hash validation is required before any retraining.
- **RR-007** (Protected attribute capture at scoring): Conditional on deployment architecture.

These risks are acceptable for initial deployment only if the deployment conditions in `docs/07_deployment_recommendation.md` are satisfied.
