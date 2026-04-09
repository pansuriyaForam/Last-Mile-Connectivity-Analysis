# 🚇 Hyderabad Metro Red Line — Last-Mile Connectivity Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![GeoSpatial](https://img.shields.io/badge/GeoSpatial-GeoPandas-green)
![Optimization](https://img.shields.io/badge/Optimization-MCLP-orange)

---

## 📌 Overview

This project builds a **data-driven pipeline** to analyze and improve last-mile connectivity along the  
**Hyderabad Metro Red Line (Miyapur ↔ LB Nagar)**.

It focuses on a simple but important question:

> **Which stations are poorly connected — and where should limited feeder resources be deployed first?**

The approach combines:
- multimodal transit data (Metro + Bus + MMTS + Feeder)
- time-based analysis (morning / midday / evening)
- an optimization model for feeder placement
- route-level suggestions for practical planning

---

## ⚙️ What the System Does

### 1. Connectivity Analysis
- Computes a **Last-Mile Connectivity Index (LMCI)** for each station  
- Captures variation across **time windows**  
- Identifies **transit deserts** (low-connectivity zones)

---

### 2. Multimodal Integration
- Combines:
  - Metro (HMRL)
  - Bus (TGSRTC)
  - MMTS suburban rail
  - Feeder services (reconstructed)

- Includes basic transfer logic and penalties

---

### 3. Optimization (MCLP)
- Uses a **budget-constrained coverage model**
- Prioritizes underserved areas using **equity weighting**
- Suggests where new feeder stops should be placed

---

### 4. Route-Level Extension
- Identifies demand clusters (DBSCAN)
- Generates candidate feeder routes:
  - direct metro links  
  - transfer-based routes  
  - circular micro-feeders  
- Ranks routes using multi-criteria scoring

---

### 5. Evaluation & Validation
- Baseline comparison (static, unimodal, standard MCLP)
- Ablation study (removing key components)
- Sensitivity to modeling assumptions

---

## 🧮 Core Idea (Simplified)

### LMCI (Connectivity Score)

```

LMCI = 10 × [Density + Frequency + Walkability]

````

- Higher LMCI → better connectivity  
- LMCI < 4 → considered a transit desert  

---

### Optimization Goal

> Select **k feeder stops** such that coverage of underserved areas is maximized.

---

## 📊 Outputs

The notebook generates:

- 📈 LMCI heatmaps (time-based)
- 📊 Transit desert classification
- 📉 Coverage curves & marginal gains
- 🧪 Ablation study results
- 🗺️ Interactive multimodal map
- 🛣️ Feeder route suggestions

All outputs are saved in the `visuals/` folder.

---

## 📂 Repository Structure

```text
├── Data/
│   ├── hmrl/        # Metro GTFS
│   ├── tgsrtc/      # Bus GTFS
│   ├── mmts/        # MMTS GTFS
│   └── feeder/      # Reconstructed feeder GTFS
│
├── visuals/         # Generated outputs (plots + maps)
├── cache/           # OSM + route cache (auto-created)
│
├── Last_Mile_Connectivity_Hyderabad_RedLine.ipynb
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

```bash
pip install pandas geopandas matplotlib folium scipy shapely osmnx
```

3. Run the notebook sequentially

---

### Notes

* Missing GTFS data → pipeline still runs with fallback handling
* OSM routing is optional (enabled by config)
* First OSM run may take longer due to graph download

---

## ⚠️ Limitations

* Uses scheduled GTFS data (not real-time)
* Demand is approximated (no census / ridership data)
* Walking distances partly approximated
* Suggested routes/stops are not field-validated

👉 This is a **decision-support tool**, not a final deployment plan.

---

## 💡 Key Takeaways

* Connectivity varies significantly across time
* Many stations remain underserved even during peak hours
* Equity-based optimization shifts focus toward peripheral zones
* Route suggestions help translate analysis into actionable planning

---

## 📈 Potential Applications

* Urban transit planning
* Feeder network design
* Accessibility analysis
* Smart city data systems

---

## 🙌 Acknowledgment

This project is part of an **AI/ML-driven urban mobility analysis initiative**,
focused on applying data science to real-world transportation problems.

---

## 📬 Contact / Next Steps

If you're working on:

* urban mobility
* transport optimization
* geospatial analytics

Feel free to connect or collaborate 🚀
