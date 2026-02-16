# AI Problem Framing & System Requirements

The objective here is to minimize entropy between product requirements and technical implementation. Rigorous problem framing prevents over-engineering and ensures that the chosen architecture aligns with real-world production constraints [1].

---

## 1. Architectural Framing: Mapping the Stochastic Objective

Before selecting a model, we must define the mathematical nature of the task:

* **Task Archetype:** Is the problem one of **Discrimination** (classification/detection), **Generation** (distribution synthesis), or **Ranking** (optimizing relative relevance)? Clear task formalization is a prerequisite for correct architectural modeling in AI systems [1].

* **Inductive Bias:** What domain knowledge can we inject into the architecture? (e.g., spatial invariance for Vision via CNNs/ViTs, or temporal dependencies for time-series). Architectural inductive bias constrains hypothesis space and directly affects generalization behavior [1].

* **Determinism vs. Probability:** Does the system tolerate "hallucination"? If the product requires 100% repeatability (e.g., financial tax calculation), AI is an architectural error. If the problem involves high-dimensional ambiguity (e.g., natural language understanding), a probabilistic model is mandatory. Large Language Models (LLMs) and Multimodal LLMs (MLLMs) inherently operate through probabilistic token prediction and distribution modeling [2][3].

---

## 2. AI Justification & Baseline (The Heuristic Test)

Every AI system must justify its operational cost (VRAM/Inference latency) within the broader system architecture [1].

* **Heuristic Baseline:** What performance can be achieved with a simple `if-else` block, a Regex, or basic statistical lookups? Empirical studies in AI requirements modeling emphasize evaluating classical baselines before adopting complex learning architectures [1].

* **ML as a Last Resort:** AI should be implemented only when the complexity of the data manifold exceeds the capacity of procedural logic. This reflects established system-engineering practices in AI deployment pipelines [1].

---

## 3. Data Topology & Operational Context

Engineers must treat data as a raw **Signal** subject to physical and logical constraints [1].

* **Signal Analysis:** Is the data stationary or subject to **Covariate Shift**? Under distribution shift, static models degrade when deployment data diverges from training distribution [1].

* **User Context & Latency Budget:** Is the inference occurring at the **Edge** (limited VRAM/Battery) or via a **Cloud Cluster**? We define a **P95/P99 Latency Target**, which dictates the need for quantization (INT8), pruning, or specific inference engines (TensorRT/ONNX). Production ML research highlights latency-constrained optimization as a core MLOps concern [4].

* **Bayes Error Rate:** What is the theoretical performance ceiling? If human experts only agree 85% of the time, a model aiming for 99% accuracy is simply overfitting to noise. This aligns with classical statistical learning theory constraints [1].

---

## 4. IP Sovereignty & Asset Strategy (The Moat)

In modern AI engineering, value increasingly resides in **Weights** and **Data**, not just code [2].

* **Weight Sovereignty:**  
   * **Rental (API-based):** Using GPT-4/Claude accelerates time-to-market but offers zero IP accumulation and introduces vendor lock-in risk.  
   * **Owned (Open Weights):** Fine-tuning open-weight models (e.g., LLaMA, Mistral, Stable Diffusion) enables proprietary checkpoint ownership and system-level customization [5].

* **Licensing & Provenance:** Training data must be audited for restrictive licenses to avoid legal contamination and ensure commercial viability. This issue has become central in generative AI system governance [2].

---

## 5. Bias Mitigation & Safety Guardrails

Bias is a technical failure of generalization, not merely a social concern [4].

* **Fairness Metrics:** We define group-based metrics such as **Demographic Parity** or **Equalized Odds**. A model that is 95% accurate globally but 50% accurate for a specific sub-population represents systemic risk [1].

* **Adversarial Robustness:** How does the system handle Out-of-Distribution (OOD) inputs or malicious prompt injections? Robustness is a foundational component of trustworthy ML systems in production [4].

* **Safety Layer:** Implementation of inference-time guardrails (e.g., Llama Guard) enforces product safety policies in generative AI systems [4].

---

## 6. Model Strategy Decision Matrix

| Family | Inductive Bias | Compute Density | Ownership Potential |
| :--- | :--- | :--- | :--- |
| **Classical ML** | High (Feature Eng) | Low (CPU) | Full |
| **Transformers** | Sequence/Global | High (GPU/HBM) | Partial to Full |
| **Diffusion** | Local Noise/Iterative | Very High (VRAM) | Full |
| **Foundation API** | Generalist | Zero (Managed) | None (Rental) |

Transformer-based systems have become dominant in large-scale language and multimodal modeling [3], while diffusion architectures have emerged as state-of-the-art in generative image modeling [5]. Retrieval-augmented API-based architectures provide practical deployment alternatives under compute constraints [6].

---

# References

[1] R. Liao et al., *Requirements Elicitation and Modelling of Artificial Intelligence Systems: An Empirical Study*, 2022. [Link](https://arxiv.org/abs/2203.15509)  

[2] D. Hendrycks et al., *A Systems Perspective on Generative AI*, 2023. [Link](https://arxiv.org/abs/2302.07236)  

[3] H. Zhang et al., *From Multimodal LLMs to Human-Level AI: Modality, Instruction, Reasoning and Beyond*, 2023. [Link](https://arxiv.org/abs/2301.11275)  

[4] B. Biggio & F. Roli, *Security and Robustness in Machine Learning*, 2018. [Link](https://arxiv.org/abs/1801.02911)  

[5] T. Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, 2022. [Link](https://arxiv.org/abs/2112.10752)  

[6] P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2021. [Link](https://arxiv.org/abs/2005.11401)  
