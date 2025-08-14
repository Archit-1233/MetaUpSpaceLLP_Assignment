# 🤖 Multi-Agent Intelligent Recruitment System (MAIRS)

## 📌 Overview
This project is a prototype for the **AI Intern Hiring Task: Multi-Agent Recruitment System**.  
It automates candidate profiling, assessment creation, soft-skill evaluation, and market analysis through **four specialized AI agents**, all integrated into a **Streamlit-based interface**.

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






