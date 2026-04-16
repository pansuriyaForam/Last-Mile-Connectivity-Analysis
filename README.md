# 🚇 Hyderabad Metro Red Line — AI-Driven Last-Mile Connectivity Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![GeoSpatial](https://img.shields.io/badge/GeoSpatial-GeoPandas-green)
![Optimization](https://img.shields.io/badge/Optimization-MCLP-orange)
![Status](https://img.shields.io/badge/Status-Research%20Grade-success)

---

## ⚡ TL;DR

> A **research-grade, end-to-end system** that diagnoses and fixes last-mile connectivity gaps in urban metro systems using
> **GTFS + Geospatial Modeling + Optimization + Route Design**

📍 Case Study: Hyderabad Metro Red Line
📊 Result: **+8.3% demand coverage uplift**, **+15.5% transit desert improvement** over standard models

---

## 🧠 Why This Project Exists

Most metro systems fail **not because of infrastructure — but because of access.**

You can build a ₹20,000 crore metro…
…but if people can’t reach the station → it’s useless.

This project answers:

> ❓ *Where is connectivity failing?*
> ❓ *When does it fail?*
> ❓ *How do we fix it optimally under constraints?*

---

## 🚀 What Makes This Different

This is **NOT** another “EDA + ML notebook”.

This is a **decision-support system** with:

### 🔹 Temporal Intelligence

* Captures **morning / midday / evening variability**
* Uses **worst-case LMCI** → no masking of peak-hour failures

### 🔹 Equity-Aware Optimization

* Traditional models → serve high-demand areas
* This model → **prioritizes underserved regions**

### 🔹 Multimodal Thinking

* Metro + Bus + MMTS + Feeder + Walkability
* Real-world system modeling (not toy datasets)

### 🔹 From Analysis → Action

* Not just “insights”
* Generates:

  * ✅ feeder stop locations
  * ✅ ranked feeder routes
  * ✅ deployable strategies

---

## 🧮 Core Engine

### 📊 Last-Mile Connectivity Index (LMCI)

A composite accessibility metric:

* Stop density
* Service frequency (time-aware)
* Walkability

👉 Used to classify **Transit Deserts**

---

### 🎯 Optimization Model

> **Equity-Weighted Maximum Coverage Location Problem (MCLP)**

* Budget constraint: `k feeder stops`
* Objective: maximize **weighted demand coverage**
* Bias: **low-LMCI zones get higher priority**

---

### 🧠 Key Insight

> Removing equity weighting → system collapses

| Model Variant | Coverage  | Desert Coverage |
| ------------- | --------- | --------------- |
| Full Model    | **71.6%** | **73.1%**       |
| Standard MCLP | 62.7%     | 57.0%           |

📉 That’s a **massive failure of naive optimization**

---

## 📊 Results (Real Findings)

### 🚨 Transit Desert Reality

* Morning: **18 stations**
* Midday: **14 stations**
* Evening: **19 stations**

👉 Connectivity is a **peak-hour failure problem**

---

### 📈 Coverage Performance

* Full model consistently dominates baselines
* Diminishing returns observed after k ≈ 7–8
* Still **>1% marginal gain** per stop → efficient allocation

---

### 🏆 Top Strategy Insight

> **MMTS-linked feeder routes outperform direct metro-only routes**

Translation:
👉 *Integration beats isolation*

---

## 🗺️ System Outputs

This pipeline generates:

* 📊 LMCI rankings & heatmaps
* 📉 Transit desert classification
* 📈 Coverage curves (MCLP vs variants)
* 🧪 Ablation studies
* 🛣️ Ranked feeder routes
* 🗺️ Spatial network visualizations

---

## 🧱 Architecture Overview

```
GTFS + OSM Data
        ↓
Preprocessing & Cleaning
        ↓
Multimodal Network Graph
        ↓
Temporal Frequency Modeling
        ↓
LMCI Computation
        ↓
Transit Desert Detection
        ↓
Equity-Based MCLP Optimization
        ↓
Route Generation (DBSCAN + Heuristics)
        ↓
Evaluation (Coverage + Ablation)
```

---

## 📂 Repository Structure

```bash
├── data/
│   ├── hmrl/
│   ├── tgsrtc/
│   ├── mmts/
│   └── feeder/
│
├── visuals/              # All plots & maps
├── cache/                # OSM cache
│
├── notebook/
│   └── main.ipynb
│
├── report/
│   └── Project_Report.docx
│
└── README.md
```

---

## 🧪 Reproducibility

```bash
pip install pandas geopandas matplotlib folium scipy shapely osmnx scikit-learn
```

Run notebook sequentially.

⚠️ Notes:

* OSM graph download → slow on first run
* Missing GTFS → fallback logic included
* Fully deterministic (seeded pipeline)

---

## ⚠️ Limitations

* No real-time traffic
* Demand is approximated
* Some walking distances approximated
* Not field validated

👉 This is a **planning intelligence layer**, not deployment

---

## 💡 What You Should Take Away

* Accessibility ≠ infrastructure
* Peak hours reveal hidden failures
* Equity-aware optimization is **non-negotiable**
* Multimodal integration beats isolated planning
* Data → Insight → Optimization → Action is the real pipeline

---

## 🧭 Real-World Applications

* 🏙 Urban transport planning
* 🚍 Feeder network design
* 📊 Smart city analytics
* 📍 Accessibility mapping
* 🚦 Policy simulation

---

## 🔥 Future Roadmap

* Real-time GTFS integration
* Reinforcement learning for dynamic routing
* Demand prediction using ML
* Deployment as a web-based decision tool

---

## 🤝 Let’s Connect

If you’re working on:

* Urban mobility
* Optimization systems
* Geo-AI
* Smart cities

👉 This is exactly the space I’m building in.

---

## ⭐ Final Note

If this project made you think:

> “Damn… this is actually useful”

Give it a ⭐ — it helps more than you think.
