# 🔬 Research & Experimental Iterations Log

This directory preserves the diagnostic and exploratory research scripts developed during the project's development cycle. They chronicle the progression from initial data profiling to the final multi-font generalization pipeline.

---

## 📌 Milestone Overview

| Stage | Script(s) | Focus & Findings |
| :--- | :--- | :--- |
| **Data Profiling** | `inspect_dataset.py`, `inspect_image_quality.py`, `check_duplicate_images.py` | Profiled 3,330 source images; identified extreme class sparsity (~7 images/class across 432 categories). |
| **Label Alignment** | `create_gujarati_labels.py`, `verify_labels.py`, `check_label_alignment.py` | Formulated Unicode mappings for both the 432-class grid and the 47 core character roots. |
| **Phase 1 Baseline (432 Classes)** | `train_character_model.py`, `diagnose_model.py`, `inspect_training_history.py` | Demonstrated severe underfitting (0.3% validation accuracy) due to insufficient samples per class. |
| **Phase 2 Root Isolation (47 Classes)** | `prepare_core_dataset.py`, `train_core_model.py`, `test_core_prediction.py` | Restructured the objective to 47 fundamental characters, raising accuracy to 26.4%. |
| **Phase 3 Production (Multi-Font)** | Moved to `src/train_robust_model.py` | Synthesized 10,000+ variations across 5 font families with morphological pen thickness and shear, achieving **82.98% validation accuracy** on genuine human handwriting. |

---

## 🚀 Active Production Pipeline
For the final, working implementation, refer directly to the root source directory:
- **Preprocessing:** `src/preprocessing.py`
- **Word Segmentation:** `src/segmentation.py`
- **Training Engine:** `src/train_robust_model.py`
- **Web UI Application:** `app.py`
