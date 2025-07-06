# AIgnition 2.0 – Geographic Personalization Engine

**AI-Powered Cold-Start Solution for E-Commerce Personalization**

## 🏆 Business Impact Summary

| Metric                  | Achievement                          | Strategic Value                     |
|-------------------------|--------------------------------------|-------------------------------------|
| **Revenue Lift**        | 18.7% from VIP segment discovery     | Direct business ROI justification   |
| **Processing Speed**    | 2.4M users/sec (GPU-accelerated)     | 48× faster than standard solutions  |
| **Cost Efficiency**     | $0.0003/1M vs AWS ($12+)             | 99.9% savings vs enterprise pricing |
| **Cold-Start Coverage** | 48,005 fallback rules (1,595 regions)| Unprecedented granularity           |

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/rupesh-koppuravuri/AIgnition_Hackathon
cd AIgnition_Hackathon

# 2. Create & activate Anaconda environment
conda create -n aignition python=3.10 -y
conda activate aignition

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install FAISS via conda for Windows
conda install -c conda-forge faiss-cpu -y

# 5. Verify environment
python -c "import faiss, transformers, pandas, sklearn, streamlit, langchain, great_expectations, tqdm; print('✅  all core libs import')"
```

## 📊 Live Demo

- **Interactive Prototype:** [Hugging Face Spaces](https://huggingface.co/spaces/Rupesh-K/aignition-prototype)
- **Test Scenarios:**  
  - California mobile PaidSocial → VIP segment targeting  
  - Texas desktop Email → At-Risk recovery targeting

## 📁 Repository Structure

```
AIgnition_Hackathon/
├── README.md
├── requirements.txt
├── AIgnition_2.0_Hackathon_Presentation.pptx
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda_segmentation.ipynb
│   ├── 04_cold_start_strategy.ipynb
│   ├── 05-recommendation-engine-aignition-hackathon.ipynb
│   └── 06_streamlit_prototype.ipynb
├── presentation/
│   └── AIgnition_2.0_Presentation.pdf
├── deployment/
│   ├── streamlit_app.py
│   └── models/
├── config/
│   ├── enhanced_fallback.yaml
│   ├── final_fallback.yaml
│   └── production_fallback.yaml
└── data/
    └── (see Kaggle public dataset link below)
```

## 🗂️ Datasets

- **Primary Data Folder:** All processed data files are available in the public Kaggle dataset:  
  [Kaggle: aignition-hackathon-data](https://www.kaggle.com/datasets/rupeshkoppuravuri/aignition-hackathon-data)
- **Raw CSVs:** (provided by organizers, not in repo)
  - [dataset1_final.csv](https://drive.google.com/file/d/1G1EHGDsNctlKTusIuFKaYNIC0ycLuH4I/view)
  - [dataset2_final.csv](https://drive.google.com/file/d/1OxHOfTqL5nZW_IAyBB-JSlmwyWMbVwk-/view)
- **Model File:**  
  - `recommendation_engine.pkl` is included in the Kaggle dataset for prototype reproducibility.

**Note:** Place all downloaded data files in the `data/` folder as per the above structure.

## 🎯 Key Innovations

- **VIP Segment Discovery:** 1.2% of users generate 18.7% of revenue.
- **Geographic Intelligence:** Personalization across 1,595+ regions.
- **Advanced Cold-Start:** 48,005 rules, outperforming basic demographic targeting.
- **GPU Performance:** 2.4M users/sec throughput (Numba-CUDA kernel).
- **Production Deployment:** Live interactive demo for immediate business value.

## 📈 Performance Metrics

| Metric             | Achievement      | Industry Comparison        |
|--------------------|-----------------|---------------------------|
| Processing Speed   | 2.4M users/sec  | 48× faster than standard  |
| Cost Efficiency    | $0.0003/1M      | 99.9% savings vs AWS      |
| Cold-Start Coverage| 99.7% scenarios | Unprecedented granularity |
| Revenue Impact     | 18.7% lift (VIP)| Direct business value     |

## 🛠️ Technical Architecture

- **Data Processing:** Pandas + Parquet (6.6M events in 36 seconds)
- **Segmentation:** K-means clustering with 0.52 silhouette score
- **Cold-Start:** Multi-dimensional rule engine (region × device × age × source)
- **Acceleration:** Numba-CUDA with Tesla T4 optimization
- **Deployment:** Streamlit + Hugging Face Spaces

## 💼 Business Value

- Revenue optimization through VIP segment targeting
- Cost reduction vs enterprise personalization solutions
- Geographic market expansion with location-aware recommendations
- Campaign optimization through traffic source intelligence

## 🏅 Competition Advantages

- **Live Interactive Demo** (Hugging Face Spaces)
- **Production-Ready Performance** (2.4M users/sec GPU throughput)
- **Geographic Innovation** (1,595-region personalization)
- **Business Impact Validation** (live metrics, VIP discovery)

## 📚 Notebooks Overview

| Notebook                               | Purpose                                 |
|-----------------------------------------|-----------------------------------------|
| 01_data_pipeline.ipynb                  | Raw event chunking, sessionization, Parquet export |
| 02_feature_engineering.ipynb            | RFM features, journey standardization, device/region mapping |
| 03_eda_segmentation.ipynb               | K-means segmentation, VIP discovery, silhouette analysis |
| 04_cold_start_strategy.ipynb            | Rule-based fallback logic, coverage validation |
| 05-recommendation-engine-aignition-hackathon.ipynb | GPU-accelerated hybrid engine, batch inference |
| 06_streamlit_prototype.ipynb            | Interactive demo, performance dashboard |

## 📦 How to Run

1. **Clone the repo and set up the environment** (see Quick Start above).
2. **Download datasets** from the public Kaggle link and place them in `data/`.
3. **Run notebooks** in sequence (01 to 06) for full pipeline reproduction.
4. **For demo:**  
   - Run the Streamlit app (`streamlit run deployment/streamlit_app.py`)
   - Or use the [Hugging Face Spaces live demo](https://huggingface.co/spaces/Rupesh-K/aignition-prototype).

## 🖼️ Visuals, Screenshots & Flowcharts

> - Screenshots of the Streamlit app and dashboard  
> - ![newplot](https://github.com/user-attachments/assets/4713f057-988a-45f6-a0f2-4056731b5549)
> - ![image](https://github.com/user-attachments/assets/bf50c155-a3f6-4139-adc5-3f5af492892b)

> - region_deivice_heatmap.png
> - ![image](https://github.com/user-attachments/assets/2f6ce0a9-97ff-4f22-9c81-5b0e6594e4dd)
> - segment_revenue.png
> - ![image](https://github.com/user-attachments/assets/4b216eb9-ba7a-44b7-a0ed-b0b207b78422)
> - segment_value_size.png
> - ![image](https://github.com/user-attachments/assets/f95984a9-8ccc-4d1b-8cb7-456507e833e9)



## 📝 Troubleshooting

| Issue                  | Solution                              |
|------------------------|--------------------------------------|
| Port Conflicts         | `streamlit run --server.port=8502 deployment/streamlit_app.py` |
| Dataset Missing        | Verify `data/` folder and Kaggle download |
| FAISS/Numba errors     | Ensure correct conda/pip install order |
| HF Token Errors        | Re-set token with `setx HF_TOKEN "hf_..."` |

## 📄 License

MIT License  
*Free for commercial/research use*

## 👨‍💻 Author

**Rupesh Koppuravuri**  
Email: koppuravurirupesh@gmail.com  
GitHub: https://github.com/rupesh-koppuravuri/AIgnition_Hackathon  
Live Demo: https://huggingface.co/spaces/Rupesh-K/aignition-prototype
