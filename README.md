# 🤖 Multi-Agent Intelligent Recruitment System (MAIRS)

## 📌 Overview
This project is a prototype for the **AI Intern Hiring Task: Multi-Agent Recruitment System**.  
It automates candidate profiling, assessment creation, soft-skill evaluation, and market analysis through **four specialized AI agents**, all integrated into a **Streamlit-based interface**.
## Please check the deployed streamlit app here : 
https://archit-1233-metaupspacellp-assignment-full-app-x2ty0q.streamlit.app/
## 🎥 Demo Video

[![Watch the video](https://github.com/user-attachments/assets/851874ad-a27d-4bb9-8240-8979a76bda3c)](https://drive.google.com/file/d/1Y6ZCTS3m-auHa3s_p42D3-hOo-XId0BB/view?usp=sharing)


### Screenshots:

## Summary
<img width="1388" height="843" alt="image" src="https://github.com/user-attachments/assets/851874ad-a27d-4bb9-8240-8979a76bda3c" />
### Skills
<img width="1423" height="821" alt="image" src="https://github.com/user-attachments/assets/3b8c7b6c-c9ff-4109-93c0-a8d4877168d0" />
## Assessment
<img width="1117" height="643" alt="image" src="https://github.com/user-attachments/assets/61a4d56a-95eb-4b64-b404-2de4f5e98c23" />
## Behavioral
<img width="1321" height="801" alt="image" src="https://github.com/user-attachments/assets/36e2e193-be60-4d04-8d60-1f841ff9874d" />
## Market Analysis
<img width="1156" height="850" alt="image" src="https://github.com/user-attachments/assets/35d27db7-e526-4ff6-a9c6-d2ebbb0a969b" />
## Raw json
<img width="915" height="782" alt="image" src="https://github.com/user-attachments/assets/b398114a-6d05-4849-8d96-4a756d6cbce1" />



The system demonstrates how **AI can enhance efficiency, fairness, and insight** in the hiring workflow.

---

## 🧠 Core Agents and Their Functions

### 1️⃣ Intelligent Candidate Profiler Agent
- Parses **synthetic candidate profiles** (LinkedIn/GitHub style).
- Extracts **technical skills** & **work experience**.
- Assigns **confidence scores** using mention frequency + recency boost.
- Summarizes **career progression**.
- Outputs a **Talent Intelligence Report**.

---

### 2️⃣ Adaptive Technical Assessment Designer Agent
- Generates **custom technical challenges** tailored to the candidate’s skills and role.
- Creates an **Evaluation Framework**:
  - Problem-solving: 40%
  - Code quality: 30%
  - Communication: 30%
- Adds a **Bias Mitigation Protocol** to guide evaluators toward fair, objective reviews.

---

### 3️⃣ Behavioral & Cultural Fit Analyzer Agent
- Analyzes **simulated interview text** to detect:
  - Collaboration
  - Problem-solving
  - Communication
- Avoids demographic and affinity biases by **focusing only on behavioral signals**.

---

### 4️⃣ Market Intelligence & Sourcing Optimizer Agent
- Processes **synthetic market data** to find:
  - Talent demand trends
  - Compensation benchmarks
  - Best sourcing channels
- Produces an **actionable market report**.

---

## 🏗 System Architecture
[Streamlit UI]
│
├── Candidate Profiler Agent
├── Assessment Designer Agent
├── Behavioral Analyzer Agent
└── Market Intelligence Agent

All agents run locally and are built with **Python**, **NLP utilities**, and **free-tier AI models**.

---

## 🛠 Data Simulation
To ensure **privacy compliance**, all input data is **synthetic**:
- **Profiles**: Fake GitHub projects, LinkedIn-like summaries, work histories.
- **Interview Text**: AI-generated Q&A interactions.
- **Market Data**: Simulated salary ranges, skill demand, and sourcing platform stats.

---

## ⚙️ Setup Instructions

## 1. Clone Repository
```bash
git clone https://github.com/yourusername/multi-agent-recruitment.git
```
### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
### 3. Install Requirements
``` bash
pip install -r requirements.txt
``
###  4. Run Application
``` bash
streamlit run app.py

```

## 💡 Design Trade-offs & Assumptions

Free-tier compliance: Avoided paid APIs; used regex + keyword extraction.
Confidence scoring: Used exponential decay to balance frequency and recency.
Synthetic data realism: Balanced realism with privacy.
UI clarity: Focused on explainable outputs over heavy styling.

 ## Bias Mitigation

Ignore demographic indicators in processing.
Evaluate skills only, not personal traits.
Apply consistent rubrics for all candidates.
Provide transparent, evidence-based reports.










