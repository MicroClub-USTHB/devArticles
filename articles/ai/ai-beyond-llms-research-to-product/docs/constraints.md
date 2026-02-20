# Performance & Operational Constraints

In modern AI systems, constraints are not merely limitationsbut rather the primary drivers of architectural selection [1]. The user provided document emphasizes defining boundaries across three critical axes: **Compute Economics**, **Temporal Performance**, and **Risk Governance**.

---

## 1. Inference & Temporal Constraints

The "User Experience" of an AI system is mathematically defined by its latency distribution.

- **Latency Thresholds (P95/P99):**
  - **Interactive (RAG/Agentic):** For interactive applications like Retrieval-Augmented Generation (RAG) systems or AI agents, a target latency of less than 500ms for first-token generation is crucial. This necessity drives the adoption of **Streaming Architectures** and potentially **Speculative Decoding** to ensure a responsive user experience [2] [3] [4]. RAG systems, for instance, combine the generative capabilities of Large Language Models (LLMs) with information retrieval to enhance accuracy and contextuality, where timely response generation is critical [5]
  - **Asynchronous (Batch/ETL):** ​​ For batch processing or Extract, Transform, Load (ETL) operations, throughput is prioritized over latency. Optimization efforts shift towards techniques like ​ **FlashAttention** and maximizing batch size to saturate GPU High Bandwidth Memory (HBM) [6]. This approach is common in scenarios where large volumes of data are processed offline, and immediate feedback is not required.
- **Throughput (RPS/RPM):**
  - Defining the expected Requests Per Second (RPS) or Requests Per Minute (RPM) is essential. High-throughput requirements can necessitate a shift from slow **Autoregressive models** to more efficient **Distilled/Quantized models** or the use of **Heuristic baselines** for pre-filtering [6]. This is particularly relevant for large-scale deployments where computational efficiency is a major concern.

---

## 2. Resource & Compute Feasibility

The available hardware resources directly dictate the feasible scale and complexity of the AI model.

- **VRAM & Memory Bandwidth:** This is a primary bottleneck for modern LLMs and diffusion models [6].
  - **Constraint:** If deployment is restricted to 24GB VRAM (e.g., NVIDIA A10), this limits the choice to smaller models, typically 7B–13B parameter models, often requiring **4-bit/8-bit Quantization (bitsandbytes/GGUF)** to fit within the memory constraints [6]. This highlights a crucial trade-off between model size, performance, and hardware accessibility. Configurable foundation models, for example, aim to address these challenges by decomposing LLMs into functional modules, allowing inference with only a subset of modules and thus reducing computational requirements [6].
- **Data Provenance & Availability:**
  - **The "Cold Start" Problem:** A lack of sufficient labeled data for a supervised baseline often forces reliance on **Zero-shot/Few-shot Foundation Models** [6]. While this reduces R&D time, it typically increases API costs if external models are used [2] [4]. The quality and quantity of available data are fundamental requirements for AI system development [1].
- **Training vs. Serving Budget:**
  - ​​It is critical to distinguish between the one-time costs associated with training a model and the recurring costs of inference. A "cheap" training phase, often achieved through API fine-tuning, can lead to a more expensive inference phase over time [3]. This economic consideration influences decisions on whether to build custom models or leverage existing foundation models via APIs.

---

## 3. Risk, Safety, & Behavioral Guardrails

AI failures can be subtle and statistical rather than immediately obvious, making robust risk mitigation essential. The importance of trustworthy machine learning in production environments is increasingly recognized [7].

### A. Failure Modes & Reliability

- **Semantic Drift:** This refers to how the system handles Out-of-Distribution (OOD) queries [7]. In high-stakes domains like medical or legal applications, a ​​ **Confidence Score** or a "I don't know" fallback mechanism is required to prevent hallucinations and ensure reliability [2].
- **Systemic Bias:** Auditing for **Disparate Impact**is crucial. The objective function should not solely optimize for Global Accuracy but also for Min-Max Fairness across sensitive demographic cohorts to prevent discriminatory outcomes [1] [8]. Bias mitigation is framed as a technical failure of generalization rather than solely a social concern, requiring robust evaluation using fairness metrics [8].

### B. Legal & Compliance Fences

- **Data Residency (GDPR/HIPAA):** If Personally Identifiable Information (PII) is involved, architectural decisions are often forced towards **On-premises/VPC execution**o comply with regulations like GDPR or HIPAA. Closed APIs are only permissible if "Zero-Data Retention" (ZDR) agreements are in place [1]. Requirements engineering for AI systems faces challenges in specifying and validating requirements related to ethical implications and compliance [8].
- **IP Protection:** To maintain a competitive advantage, proprietary data must never be used to train models owned by third-party providers unless explicitly opted out [1]. This ensures IP sovereignty and prevents vendor lock-in, aligning with strategies for controlling "Weights" and "Data" assets [9].

---

## 4. Success Metrics (The KPI Alignment)

We distinguish between **Model Metrics** (Technical) and **System Metrics** (Business).

| Category     | Metric                             | Goal                                                   |
| :----------- | :--------------------------------- | :----------------------------------------------------- |
| **Model**    | F1-Score / NDCG / Perplexity       | Ensure mathematical convergence and retrieval quality. |
| **System**   | Cost-per-Inference / P99 Latency   | Ensure operational and economic sustainability.        |
| **Business** | Task Automation Rate / UX Friction | Ensure the AI actually solves the human bottleneck.    |


These metrics, as outlined in the user's document, provide a comprehensive framework for evaluating the effectiveness and success of an AI system from multiple perspectives. Model metrics focus on the technical performance of the AI, while system metrics assess operational efficiency, and business metrics measure the tangible impact on organizational goals 
[1].

---

# References


1. Khlood Ahmad, Mohamed Abdelrazek, Chetan Arora, John Grundy & Muneera Bano. (2023). *Requirements Elicitation and Modelling of Artificial Intelligence Systems: An Empirical Study*. arXiv. https://arxiv.org/abs/2302.06034 

2. Jakub M. Tomczak. (2024). *Generative AI Systems: A Systems-based Perspective on Generative AI*. arXiv. https://arxiv.org/abs/2407.11001 

3. Ioannis Tzachristas. (2024). *Creating an LLM-based AI-agent: A high-level methodology towards enhancing LLMs with APIs*. arXiv. https://arxiv.org/abs/2412.13233 

4. Sumedh Rasal (et al.). (2024). *A Multi-LLM Orchestration Engine for Personalized, Context-Rich Assistance*. arXiv. https://arxiv.org/abs/2410.10039 

5. Ayman Asad Khan & Md Toufique Hasan (et al.). (2024). *Developing Retrieval Augmented Generation (RAG)-based LLM Systems from PDFs: An Experience Report*. arXiv. https://arxiv.org/abs/2410.15944 

6. Chaojun Xiao, Zhengyan Zhang, Chenyang Song (et al.). (2024). *Configurable Foundation Models: Building LLMs from a Modular Perspective*. arXiv. https://arxiv.org/abs/2409.02877 

7. Firas Bayram & Bestoun S. Ahmed. (2024). *Towards Trustworthy Machine Learning in Production: An Overview of the Robustness in MLOps Approach*. arXiv. https://arxiv.org/abs/2410.21346 

8. Umm-e-Habiba, Markus Haug, Justus Bogner & Stefan Wagner. (2024). *How Mature is Requirements Engineering for AI-based Systems? A Systematic Mapping Study on Practices, Challenges, and Future Research Directions*. arXiv. https://arxiv.org/abs/2409.07192 

9. Gabriele Fossi, Youssef Boulaimen (et al.). (2024). *SwiftDossier: Tailored Automatic Dossier for Drug Discovery with LLMs and Agents*. arXiv. https://arxiv.org/abs/2409.15817 
