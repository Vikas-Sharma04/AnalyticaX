# 📊 AnalyticaX — Next-Gen Statistical Intelligence

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

**AnalyticaX** is a high-performance, web-based data exploration engine. It transforms raw datasets into actionable insights through a custom "Cyber-Dark" interface, bridging the gap between raw CSV/Excel files and neural-level data intelligence.

---

## 🚀 Core Features

### 🧠 Neural Search & Filtering
* **NLP Query Engine:** Filter data using natural language (e.g., `"age > 30 and city is London"`).
* **Multi-Layer Filtering:** Stack complex logic gates (Equal, Between, In-List) to drill down into specific data nodes.
* **Bypass Logic:** Toggle between global and filtered views instantly across the entire workspace.

### 🔬 Intelligence & Diagnostics
* **Smart Cleaner:** One-click "NULL FIX" (Median/Mode imputation) and "CLEAN DUPES" (deduplication).
* **Statistical Deep-Dive:** Comprehensive descriptive statistics and metadata overviews.
* **Outlier Detection:** Automated IQR (Interquartile Range) fencing with visual box-plot identification.


[Image of box plot with interquartile range outliers]


### 🎨 Visualizer & Logic Hub
* **Dynamic Chart Builder:** Render Bar, Line, Scatter, Histogram, Box, and Pie charts via Plotly integration.
* **Correlation Matrix:** Instant heatmap generation to identify linear relationships between features.


[Image of correlation matrix heatmap]

* **Grouped Analytics:** Execute SQL-like aggregations (Sum, Mean, Count) with immediate visual feedback.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Engine:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Graphics:** [Plotly Express](https://plotly.com/python/plotly-express/)
* **Styling:** Custom CSS / Inter Font / Cyber-Dark Theme

---

## 📦 Quick Start

### Setup Environment and Launch app
```bash
# Clone the repository
git clone [https://github.com/yourusername/AnalyticaX.git](https://github.com/yourusername/AnalyticaX.git)
cd AnalyticaX

# Install dependencies
pip install -r requirements.txt

# Launch App
streamlit run app.py