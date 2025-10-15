# Human Memory Decoder

This repository contains code and minimal example data accompanying the paper  
**“Reading specific memories from human neurons before and after sleep”** (*Nature Portfolio, 2025*).

The project introduces a transformer-based neural decoding framework that learns to map intracranial neural population activity (macro- and micro-electrode recordings) to conceptual memory representations.  
Each participant’s model is trained on neural data collected during movie viewing and evaluated on free-recall sessions, demonstrating that population-level neural activity encodes specific remembered concepts above chance.

---

## Contents
- **`src/`** — Data loader, model architecture, etc.
- **`scripts/`** — Scripts for decoder evaluation, statistical testing, and visualization  
- **`data/`** — Simulated demo data illustrating model inputs and outputs

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/roychowdhuryresearch/human-memory-decoder.git
cd human-memory-decoder
pip install -r requirements.txt
```

---

## Demo (Simulated Data)
To comply with patient privacy and IRB regulations, this repository provides **synthetic data** generated with `numpy.random` and `numpy.exponential` to reproduce the full analysis pipeline.  
These data match the structure and format of the real dataset but contain **no real neural recordings or identifiable information**.

### Included Simulated Files
- **`data/annotations_simulated/`** — simulated vocalization annotations formatted identically to real `.ann` files  
- **`data/spike_data/`** — simulated preprocessed neural recordings  
- **`data/8concepts_movie_label.npy`** — real label array used for training and evaluation  

### Download
All simulated data can be downloaded from the following Google Drive link: 🔗 [Drive](https://drive.google.com/drive/folders/10fumXuPgnnqy0GPoEdtVHGdQvKqYik04?usp=sharing)

### Running the demo
Run the following commands to execute the end-to-end example:
```bash
python train_and_evaluate_demo.py
```
This will:
1. Train the transformer model on simulated movie-viewing data
2. Evaluate the model on synthetic recall data
3. Create the metadata files required for visualizations

Then run:
```bash
python scripts/population_level_mcs.py
python scripts/leision_study.py
```
This will:
This will generate the visualizations corresponding to the decoding and population-level analyses, as well as the lesion study.


---

## Citation
If you use this code, please cite:  
> *Ding, Y.*, et al. (2025). **Reading specific memories from human neurons before and after sleep.** *Nature Portfolio.*

---

## License
Released under the [MIT License](LICENSE).  
See the manuscript for additional methodological details.
