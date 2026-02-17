┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Data Extraction (MIMIC-IV)                               │
│  • Pull 1,000 clinical vignettes                                   │
│  • Strip all demographic mentions → create templates               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Counterfactual Generation                                │
│  • Systematic demographic substitution (race × gender × insurance) │
│  • 6 variations per case = 6,000 total prompts                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: MedGemma Inference (Batch)                               │
│  • Generate diagnoses + confidence scores + explanations           │
│  • Temperature=0.7, N=3 samples per prompt for variance          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Automated Analysis (Zero Ground Truth)                   │
│  ├─ Semantic divergence (SBERT)                                  │
│  ├─ Confidence variance analysis                                   │
│  ├─ Logical contradiction detection (self-consistency)             │
│  ├─ Epidemiological plausibility check (CDC API)                 │
│  └─ SUDO discrepancy across demographic strata                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Synthetic Validation                                       │
│  • Generate 500 synthetic cases with invariant demographics        │
│  • Re-run pipeline → measure "pure bias" signal                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Bias Heatmap + Flagged Cases + Statistical Report          │
│  • No expert review required                                       │
│  • Fully reproducible                                              │
│  • Ground truth agnostic                                           │
└─────────────────────────────────────────────────────────────────┘