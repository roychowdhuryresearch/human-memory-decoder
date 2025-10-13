# Human Memory Decoder

This repository contains code and minimal example data accompanying the paper  
**“Reading specific memories from human neurons before and after sleep”** (*Nature Portfolio, 2025*).

The project introduces a transformer-based neural decoding framework that learns to map intracranial neural population activity (macro- and micro-electrode recordings) to conceptual memory representations.  
Each participant’s model is trained on neural data collected during movie viewing and evaluated on free-recall sessions, demonstrating that population-level neural activity encodes specific remembered concepts above chance.

---

## Contents
- **`models/`** — Transformer-based decoding architectures
- **`analysis/`** — Scripts for decoder evaluation, statistical testing, and visualization  
- **`demo/`** — Example notebook for running inference on sample data  
- **`data/`** — De-identified demo data illustrating model inputs and outputs  

---

## Citation
If you use this code, please cite:  
> *Ding, Y.*, et al. (2025). **Reading specific memories from human neurons before and after sleep.** *Nature Portfolio.*

---

## License
Released under the [MIT License](LICENSE).  
See the manuscript for additional methodological details.
