# Claims Log
_Last updated: {{YYYY-MM-DD}}_

This document tracks:
- Evidence-backed mechanistic claims
- Design rules extracted from experiments
- Robust negative results

Every claim here should be:
- Linked to at least one interp day
- Supported by a metric and a figure

---

## 1. Core Mechanistic Claims
_(These are your strongest results.)_

### C{{NN}} — {{Short declarative title}}
- **Date established:** {{YYYY-MM-DD}}
- **Confidence (0–1):** {{ e.g., 0.7 }}
- **Supported by:** {{ day15.md, day16.md }}

**Claim:**  
> {{ Clear, declarative, evidence-backed statement }}

**Evidence summary:**  
- {{ Metric: what changed and by how much }}
- {{ Figure: filename + what it shows }}

**Scope & limits:**  
- {{ Which models, layers, tasks this applies to }}
- {{ Where it might break }}

---

## 2. Design Rules / Heuristics
_(Portable insights that guide future work.)_

### R{{NN}} — {{Rule of thumb}}
- **Date added:** {{YYYY-MM-DD}}

**Rule:**  
> {{ e.g., "Head ablations give higher signal than MLP ablations for token-copy tasks in small transformers." }}

**Derived from:**  
- {{ Which claims or experiments motivated it }}

**Operational impact:**  
- {{ How this changes how you design experiments }}

---

## 3. Robust Negative Results
_(These are extremely valuable and often publishable.)_

### N{{NN}} — {{What did NOT matter}}
- **Date established:** {{YYYY-MM-DD}}

**Negative result:**  
> {{ e.g., "Ablating early-layer MLPs does not significantly affect semantic token prediction." }}

**Evidence summary:**  
- {{ Metrics showing negligible effect }}
- {{ Figures showing flat response }}

**Why this matters:**  
- {{ What story this rules out }}

---

## 4. Claim Dependencies (Optional, Update Weekly)
_(Which claims rely on which others.)_

- C03 depends on C01, C02
- R01 derived from C01, N02

---

## 5. Paper Mapping (Optional, End of Week 3)
_(This turns directly into your paper outline.)_

- **Introduction:** C01, C02  
- **Methods:** Day 15–Day 18  
- **Results:** C01, C03, N01  
- **Discussion:** R01, R02  