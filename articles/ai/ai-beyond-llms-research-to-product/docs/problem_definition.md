# AI Problem Framing & System Requirements

The objective here is to minimize entropy between product requirements and technical implementation. Rigorous problem framing prevents over-engineering and ensures that the chosen architecture aligns with real-world production constraints.

---

## 1. Architectural Framing: Mapping the Stochastic Objective

Before selecting a model, we must define the mathematical nature of the task:

* **Task Archetype:** Is the problem one of **Discrimination** (classification/detection), **Generation** (distribution synthesis), or **Ranking** (optimizing relative relevance)?
* **Inductive Bias:** What domain knowledge can we inject into the architecture? (e.g., spatial invariance for Vision via CNNs/ViTs, or temporal dependencies for time-series).
* **Determinism vs. Probability:** Does the system tolerate "hallucination"? If the product requires 100% repeatability (e.g., financial tax calculation), AI is an architectural error. If the problem involves high-dimensional ambiguity (e.g., natural language understanding), a probabilistic model is mandatory.



---

## 2. AI Justification & Baseline (The Heuristic Test)

Every AI system must justify its operational cost (VRAM/Inference latency).
* **Heuristic Baseline:** What performance can be achieved with a simple `if-else` block, a Regex, or basic statistical lookups? If a heuristic achieves 80% of the goal with near-zero latency, the ML solution must prove marginal value exceeding its technical debt.
* **ML as a Last Resort:** AI should be implemented only when the complexity of the data manifold exceeds the capacity of procedural logic.

---

## 3. Data Topology & Operational Context

Engineers must treat data as a raw **Signal** subject to physical and logical constraints:

* **Signal Analysis:** Is the data stationary or subject to **Covariate Shift**? Static models fail if the input distribution shifts frequently.
* **User Context & Latency Budget:** Is the inference occurring at the **Edge** (limited VRAM/Battery) or via a **Cloud Cluster**? We define a **P95/P99 Latency Target**, which dictates the need for quantization (INT8), pruning, or specific inference engines (TensorRT/ONNX).
* **Bayes Error Rate:** What is the theoretical performance ceiling? If human experts only agree 85% of the time, a model aiming for 99% accuracy is simply overfitting to noise.

---

## 4. IP Sovereignty & Asset Strategy (The Moat)

In modern AI engineering, value resides in **Weights** and **Data**, not just code.

* **Weight Sovereignty:** 
   * **Rental (API-based):** Using GPT-4/Claude. Accelerates Time-to-Market but offers zero IP accumulation and creates critical vendor lock-in.
    * **Owned (Open Weights):** Fine-tuning Llama-3/Mistral/Stable Diffusion. Optimized checkpoints become proprietary intangible assets.
* **Licensing & Provenance:** 
 We must audit training data for restrictive licenses (e.g., avoiding "copyleft" contamination) to ensure the resulting model can be patented or commercialized without legal friction.

---

## 5. Bias Mitigation & Safety Guardrails

Bias is a technical failure of generalization, not just a social concern.

* **Fairness Metrics:** We define group-based metrics such as **Demographic Parity** or **Equalized Odds**. A model that is 95% accurate globally but 50% accurate for a specific sub-population is a systemic risk.
* **Adversarial Robustness:** How does the system handle Out-of-Distribution (OOD) inputs or malicious prompt injections?
* **Safety Layer:** Implementation of inference-time guardrails (e.g., Llama Guard) to enforce product safety policies.



---

## 6. Model Strategy Decision Matrix

| Family | Inductive Bias | Compute Density | Ownership Potential |
| :--- | :--- | :--- | :--- |
| **Classical ML** | High (Feature Eng) | Low (CPU) | Full |
| **Transformers** | Sequence/Global | High (GPU/HBM) | Partial to Full |
| **Diffusion** | Local Noise/Iterative | Very High (VRAM) | Full |
| **Foundation API** | Generalist | Zero (Managed) | None (Rental) |

