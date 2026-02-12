# Performance & Operational Constraints

In modern AI systems, constraints are not mere limitations; they are the primary drivers of architectural selection. We define our boundaries across three critical axes: **Compute Economics**, **Temporal Performance**, and **Risk Governance**.

---

## 1. Inference & Temporal Constraints
The "User Experience" of an AI system is mathematically defined by its latency distribution.

* **Latency Thresholds (P95/P99):** 
   * **Interactive (RAG/Agentic):** Target <500ms for first-token generation. This necessitates **Streaming Architectures** and potentially **Speculative Decoding**.
    * **Asynchronous (Batch/ETL):** Prioritize throughput over latency. Optimization focus shifts to **FlashAttention** and batch size maximization to saturate GPU HBM (High Bandwidth Memory).
* **Throughput (RPS/RPM):** 
  * We must define the expected Requests Per Second. High-throughput requirements may force a move from **Autoregressive models** (slow) to **Distilled/Quantized models** or **Heuristic baselines** for pre-filtering.



---

## 2. Resource & Compute Feasibility
The hardware "floor" dictates the model's "ceiling."

* **VRAM & Memory Bandwidth:** The primary bottleneck for modern LLMs/Diffusion models. 
    * **Constraint:** If deployment is restricted to 24GB VRAM (e.g., NVIDIA A10), we are limited to 7B–13B parameter models with **4-bit/8-bit Quantization (bitsandbytes/GGUF)**.
* **Data Provenance & Availability:** 
  * **The "Cold Start" Problem:** Do we have enough labeled data for a supervised baseline? If not, the system must rely on **Zero-shot/Few-shot Foundation Models**, increasing API costs but decreasing R&D time.
* **Training vs. Serving Budget:**
  * Distinguish between one-time training costs and recurring inference costs. A "cheap" training phase (API fine-tuning) often leads to an expensive inference phase.

---

## 3. Risk, Safety, & Behavioral Guardrails
Failure in AI is often silent and statistical rather than explicit.

### A. Failure Modes & Reliability
* **Semantic Drift:** How does the system handle Out-of-Distribution (OOD) queries? We require a **Confidence Score** or a "I don't know" fallback mechanism to prevent hallucinations in high-stakes domains (Medical/Legal).
* **Systemic Bias:** We audit for **Disparate Impact**. Our objective function must not only optimize for `Global Accuracy` but also for `Min-Max Fairness` across sensitive cohorts.

### B. Legal & Compliance Fences
* **Data Residency (GDPR/HIPAA):** If PII (Personally Identifiable Information) is involved, the architecture is forced toward **On-premises/VPC execution**. Closed APIs are only permissible if "Zero-Data Retention" (ZDR) agreements are in place.
* **IP Protection:** To maintain a competitive "Moat," proprietary data must never be used to train models owned by third-party providers unless explicitly opted out.

---

## 4. Success Metrics (The KPI Alignment)

We distinguish between **Model Metrics** (Technical) and **System Metrics** (Business).

| Category | Metric | Goal |
| :--- | :--- | :--- |
| **Model** | F1-Score / NDCG / Perplexity | Ensure mathematical convergence and retrieval quality. |
| **System** | Cost-per-Inference / P99 Latency | Ensure operational and economic sustainability. |
| **Business** | Task Automation Rate / UX Friction | Ensure the AI actually solves the human bottleneck. |



