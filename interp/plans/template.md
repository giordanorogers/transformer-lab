# Interp Session Planner — Day {{NN}}
_Date: {{YYYY-MM-DD}}_

---

## 0. What I Took In Today (Upstream Context)
_Write freely. No boxes to “get right.”_

### 0.1 Theory / Math / Concepts
- {{ What ideas, equations, or intuitions did I encounter today? }}
- {{ What felt unclear, subtle, or powerful? }}

### 0.2 Implementation / Systems
- {{ What did I build, modify, or debug today? }}
- {{ What part of the codebase now feels more “real” to me? }}

---

## 1. Open Connection Brainstorm (Not a Decision Yet)
_This is a **divergent thinking** section. List multiple weak connections without judging them._

- “If today’s ideas were visible inside a model, they might show up as…”
  - {{ a head? }}
  - {{ a direction in residual space? }}
  - {{ a training artifact? }}
  - {{ a stability issue? }}
  - {{ a geometry effect? }}

- “If today’s implementation mattered mechanistically, it might be because…”
  - {{ it routes information }}
  - {{ it compresses features }}
  - {{ it stabilizes gradients }}
  - {{ it gates computation }}
  - {{ it aligns representations }}

_Write at least 3 candidate mechanisms, even if they feel dumb._

---

## 2. Today’s Interp Angle (Now We Converge)
_From the brainstorm above, pick **one thread** to pull on—not because it’s perfect, but because it’s testable today._

> **Chosen angle for today:**  
> {{ e.g., “Information routing through a specific attention head.” }}

---

## 3. Question Framed as Curiosity (Not Yet a Hypothesis)
_Not everything has to start as a clean falsifiable claim._

> {{ “Where does X actually live in the network?” }}  
> {{ “Is there localized structure here or is it diffuse?” }}  
> {{ “Does this component matter at all?” }}

---

## 4. What I Will Actually Touch (Concrete but Open)
- **Behavior / phenomenon of interest:**  
  {{ anything: copying, induction, syntax, memorization, stability, etc. }}

- **Primary internal object(s):**  
  {{ one or two components: head, MLP, residual stream, embedding, logit space, etc. }}

This is allowed to be **vague at first** and sharpen during the session.

---

## 5. Provisional Hypothesis (Lightweight)
_Not a legal proof—just a directional belief._

> {{ “My current guess is that ________ will have some noticeable effect on ________.” }}

Optional confidence: {{ low / medium / high }}

---

## 6. Intervention Menu (Pick What Feels Most Informative Today)
_Check any that feel relevant—you’ll choose one after this._

- [ ] Ablate / zero something
- [ ] Patch activations between prompts/examples
- [ ] Add noise or rescale
- [ ] Read out with logit lens or unembedding
- [ ] Compare geometries (PCA, cosine, clustering)
- [ ] Compare across layers
- [ ] Compare across inputs
- [ ] Other: {{ }}

---

## 7. Measurement Menu (Also Open)
_I don’t have to choose yet, but I must end up with at least one number and one picture._

- Possible **numbers**:
  - {{ Δ logit }}
  - {{ Δ loss }}
  - {{ accuracy change }}
  - {{ rank shift }}
  - {{ cosine similarity }}
  - {{ variance explained }}

- Possible **visuals**:
  - {{ logit lens sweep }}
  - {{ PCA projection }}
  - {{ token trajectory }}
  - {{ ablation bar plot }}
  - {{ similarity heatmap }}

---

## 8. Session Success = One Clear Answer to One Clear Question
> “At the end of this session, I want to be able to say:  
> **__________ does / does not matter for __________ in this model.**”

---

## 9. Scope Guardrails (Still Non-Negotiable)
- ✅ One main question
- ✅ One primary component family
- ✅ One intervention type actually executed
- ✅ One metric + one figure produced
- ✅ Stop at 2.5 hours even if things get interesting

---

## 10. If I Feel Lost After 20 Minutes
_I default to a generic but always-valid probe:_

- Pick **any mid-layer attention head**
- Run **hard ablation**
- Measure **Δ logit of correct token**
- Plot **layerwise logit lens**