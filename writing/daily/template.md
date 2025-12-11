# Day {{NN}} – Daily Research Log
_Date: {{YYYY-MM-DD}}_

## 0. Metadata (5 min)
- **Start time:** {{HH:MM}}
- **End time:** {{HH:MM}}
- **Total focused work today (hrs):** {{ }}
- **Main focus today (1 sentence):**  
  {{ e.g., "Tested head 2.3’s effect on copying subject tokens." }}

---

## 1. Raw Dump – What Actually Happened (30 min)
_(No editing. No polishing. Just write.)_

### 1.1 What did I try today?
- {{ Free text – what you attempted in Blocks 2 & 3. }}

### 1.2 What broke or failed?
- {{ Bugs, dead ends, dumb mistakes, confusing results. }}

### 1.3 What surprised me?
- {{ Any behavior, metric, or pattern you didn’t anticipate. }}

### 1.4 What now seems false?
- {{ Beliefs/hypotheses that today’s work weakened or killed. }}

---

## 2. Structured Experiment Record (30 min)
_(Compress the mess above into one clean “experiment block.” If you ran multiple, pick the most important.)_

### 2.1 Setup
- **Target behavior:**  
  {{ e.g., "Next-token prediction for country → capital prompts." }}
- **Internal component:**  
  {{ e.g., "Attention head 3.2 value output." }}
- **Hypothesis (falsifiable):**  
  {{ e.g., "Head 3.2 is causally necessary for copying the correct capital token." }}
- **Intervention type:**  
  - [ ] Head ablation  
  - [ ] MLP ablation  
  - [ ] Residual patching  
  - [ ] Logit lens probe  
  - [ ] Activation swap  
  - [ ] Other: {{ }}

### 2.2 Procedure
_1–3 bullet points, concrete steps only._
- {{ Step 1 }}
- {{ Step 2 }}
- {{ Step 3 }}

### 2.3 Measurements
- **Metrics recorded (with units):**  
  - {{ e.g., "Δ logit(correct token): -2.3" }}  
  - {{ e.g., "Δ loss: +0.47" }}  
  - {{ e.g., "Top-1 → Top-4 rank shift for correct token." }}

- **Plots/figures generated (with filenames):**  
  - {{ e.g., "`interp/figures/logit_lens/day17_head32.png` – logit lens across layers." }}
  - {{ e.g., "`interp/figures/patching/day17_residual_swap.png` – patched vs baseline probs." }}

### 2.4 Observed Effect (1–3 sentences)
_Write like a result in a paper, not a diary entry._

> {{ e.g., "Ablating head 3.2 reduced the correct capital’s logit by 2.3 on average and dropped accuracy from 92% to 41% across 50 country–capital prompts, suggesting this head plays a key causal role in copying the capital token into the residual stream." }}

---

## 3. Claim / Rule Extraction (30 min)
_(This is the “turning evidence into knowledge” section.)_

### 3.1 Today’s Main Claim (pick exactly one)
Fill in ONE of these, delete the others:

- **Falsifiable claim:**  
  > {{ "When head {{L.H}} is ablated, performance on {{TASK}} drops by at least {{X}}%." }}

- **Broken/updated hypothesis:**  
  > {{ "I previously believed X, but today’s results suggest Y instead." }}

- **Design rule:**  
  > {{ "For this model family, early-layer MLP ablations barely affect semantics compared to attention head ablations." }}

### 3.2 How confident am I in this claim? (0–1)
- **Confidence:** {{ e.g., 0.3, 0.6, 0.8 }}  
- **Why this confidence level?**  
  {{ Sample size, sanity checks done, obvious caveats. }}

---

## 4. Integration with Bigger Picture
### 4.1 How does today’s result connect to previous days?
- {{ e.g., "Supports the idea that specific heads act as copy/gate units rather than diffuse behavior." }}

### 4.2 What does this suggest about transformer computation?
- {{ One short paragraph: what you now think the model is “doing.” }}

---

## 5. Plan for Tomorrow (5–10 min)
Keep this brutally concrete.

### 5.1 Next Experiment
- **Hypothesis to test:**  
  {{ }}
- **Component(s) to intervene on:**  
  {{ }}
- **Intervention type:**  
  {{ e.g., "Patch residual stream of layer 4 from prompt A into prompt B." }}
- **Success criterion:**  
  {{ e.g., "See ≥ X change in logit or accuracy for Y% of prompts." }}

### 5.2 Risks / Confusions
- {{ Anything you’re stuck on, or worried might be a timesink. }}

---

## 6. Quick Self-Check (2 min)
_(Checkboxes to keep you honest.)_

- [ ] Did I write down at least one **falsifiable hypothesis**?
- [ ] Did I log at least one **metric** (number) and one **figure** (path)?
- [ ] Did I extract **one main claim** for today?
- [ ] Did I define **one concrete thing** to test tomorrow?

---

## 7. Optional: Mental State Snapshot (totally optional)
- **Energy level (0–10):** {{ }}
- **Frustration level (0–10):** {{ }}
- **One sentence about how I feel about the work right now:**  
  {{ }}