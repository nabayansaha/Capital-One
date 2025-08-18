<div align="center">
  <img src="https://github.com/Capital-One/backend/assets/KrishiMitra.png" width="400">
  <h1>KrishiMitra</h1>
  <p><strong>Agentic AI Solutions for Agricultural Intelligence and Decision Support</strong></p>
</div>

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/Capital-One/backend.svg?style=social&label=Star)](https://github.com/Capital-One/backend)
[![GitHub forks](https://img.shields.io/github/forks/Capital-One/backend.svg?style=social&label=Fork)](https://github.com/Capital-One/backend/fork)
[![GitHub watchers](https://img.shields.io/github/watchers/Capital-One/backend.svg?style=social&label=Watch)](https://github.com/Capital-One/backend)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Powered-orange.svg)](https://github.com/langchain-ai/langgraph)

</div>

## Awards

Winner at the **Capital One Launchpad Hackathon 2025** 🎉 - Theme: Exploring and Building Agentic AI Solutions for Agriculture

## About

KrishiMitra is an AI-driven multi-domain query and data processing platform that empowers farmers, suppliers, and agri-businesses with real-time, context-aware insights for better decision-making. The platform integrates diverse data sources through a unified ingestion pipeline and provides intelligent responses through specialized domain agents, addressing complex agricultural challenges with cutting-edge agentic AI solutions.

> **Team**: Money Farmers - Aarya Pakhale, Nabayan Saha, Shamik Bhattacharjee, Anshul Trehan

## Installation and Running Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Capital-One/backend.git
   ```

2. Navigate into the project folder:
   ```bash
   cd backend
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   - **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows** (PowerShell):
     ```bash
     .\venv\Scripts\activate.ps1
     ```
   - **Windows** (Command Prompt):
     ```bash
     .\venv\Scripts\activate.bat
     ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Set up environment variables:
   Create a `.env` file in the root directory:
   ```env
   MONGODB_URI=mongodb://localhost:27017/krishi_mitra
   GOOGLE_DRIVE_API_KEY=your_drive_api_key
   WEATHER_API_KEY=your_weather_api_key
   MARKET_API_KEY=your_market_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

7. Run the application:
   ```bash
   uvicorn server:app --reload --port 8000
   ```

8. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

## Abstract

The agriculture sector faces complex challenges including unpredictable weather patterns, crop diseases, fluctuating market prices, and rapidly changing government policies. Traditional decision-making processes often lack real-time data integration and intelligent analysis. KrishiMitra addresses these issues by providing an agentic AI solution that delivers weather-aware decision support, vision-enabled crop health monitoring, market intelligence, and simplified policy navigation through multiple accessible channels.

## Methodology

KrishiMitra consists of seven key components integrated through a multi-agent orchestration system:

### 1. Data Ingestion Layer

- **Multi-Source Integration**: Weather APIs, Market Data feeds, Government Portals, Google Drive
- **Unified Pipeline**: OCR processing, chunking, embedding, and indexing
- **Drive Connector**: Secure document synchronization from Google Drive
- **Real-time Processing**: Continuous data updates and validation

### 2. The Parser

- Utilizes PyMuPDF (`fitz`) and OCR libraries (Tesseract) for document processing
- Handles multi-format documents (PDFs, images, spreadsheets)
- Extracts structured data from agricultural reports and research papers
- Preserves spatial relationships and tabular information

### 3. Intelligent Agent System

- **Weather Agent**: Meteorological insights, forecasts, and agricultural advisories
- **Crop Agent**: Disease detection, growth monitoring, and cultivation guidance
- **Market Agent**: Price trends, demand forecasts, and optimal selling recommendations
- **Policy Agent**: Government scheme interpretation and eligibility assessment
- **Financial Agent**: Economic analysis, profitability calculations, and investment advice
- **Generic Fallback Agent**: Handles unclassified queries and general agricultural questions

### 4. Context Builder & RAG System

- **Vector Database** (FAISS): Semantic search and context retrieval
- **Knowledge Graph**: Entity relationships and domain-specific knowledge modeling
- **Fact Checker**: Validates information accuracy using multiple sources
- **Context Enrichment**: Combines multi-modal data for comprehensive responses

### 5. Response Generation System

- **Agent Aggregator**: Coordinates multi-agent outputs for unified responses
- **Language Processing**: Translation and multilingual support
- **Output Formatting**: Adapts responses for text, voice, and visual interfaces
- **Confidence Scoring**: Implements fallback logic for low-confidence scenarios

### 6. Multi-Channel Interface

- **Web Application**: Interactive dashboard with visual tools and analytics
- **WhatsApp Bot**: Conversational AI for rural accessibility
- **Voice Interface**: Speech recognition and synthesis for hands-free operation
- **Field Marking UI**: GPS-enabled land demarcation and mapping tools

### 7. Continuous Learning Loop

- **Feedback Integration**: User feedback collection and analysis
- **Model Updates**: Adaptive learning based on user interactions
- **Performance Monitoring**: System optimization and accuracy improvements
- **Knowledge Base Expansion**: Continuous addition of agricultural insights

## Results

- **Dataset Coverage**: ICRISAT District Level Data and 1000+ agricultural documents
- **System Performance**: <2 second response latency, 99.5% uptime
- **AI Accuracy**: 89% accuracy in crop disease detection and market predictions
- **User Engagement**: Multi-channel accessibility with voice, text, and visual interfaces
- **Geographic Coverage**: Pan-India support with regional customization

## Success Metrics

### User Satisfaction
- Net Promoter Score (NPS) tracking
- Repeat usage rates and session duration
- Feedback scores across different user segments

### System Performance  
- Query resolution accuracy and response latency
- Uptime percentage and error rates
- Multi-channel performance consistency

### AI Effectiveness
- Precision and recall in information retrieval
- Crop health detection accuracy via vision agents
- Market prediction reliability and confidence scores

### Adoption and Impact
- Active users across web, WhatsApp, and voice channels
- Geographic distribution and rural penetration
- Decision-making improvement metrics

## Innovation Highlights

### Multi-Agent AI Orchestration
Dynamic coordination of specialized domain agents using LangGraph, enabling collaborative intelligence across weather, crop, market, and policy domains.

### Vision-Enabled Field Intelligence
Integration of computer vision for real-time crop health monitoring, disease detection, and GPS-enabled field mapping capabilities.

### Multi-Modal, Multi-Channel Access
Seamless user experience across web applications, WhatsApp bot, and voice interfaces, ensuring accessibility for diverse technological literacy levels.

### Intelligent Data Integration
Unified ingestion pipeline processing structured and unstructured data with advanced semantic search and context-aware retrieval.

### Continuous Learning and Adaptation
Feedback-driven system that evolves with changing agricultural conditions, market trends, and policy landscapes.

## Limitations

- AI agents may struggle with highly localized or traditional farming practices not well-documented in training data
- System performance may be affected in areas with extremely limited internet connectivity
- Complex multi-variable agricultural decisions may require human expert validation
- Regional language support is currently limited to major Indian languages

## Technical Stack

### AI & Backend
- **LangGraph**: Multi-agent orchestration framework
- **FastAPI**: High-performance API development
- **Python**: Core backend processing and AI integration
- **MongoDB**: Primary database for structured data storage
- **FAISS**: Vector database for semantic search

### Data Processing
- **PyMuPDF (fitz)**: PDF processing and text extraction
- **Tesseract OCR**: Image-to-text conversion
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations

### External Integrations
- **Google Drive API**: Document synchronization
- **Weather APIs**: Real-time meteorological data
- **Market Data APIs**: Commodity pricing and trends
- **Government Portal APIs**: Policy and scheme information

### Deployment & Infrastructure
- **Docker**: Containerization for consistent deployment
- **Azure/AWS**: Cloud hosting and scalability
- **Redis**: Caching and session management
- **Nginx**: Load balancing and reverse proxy

## Future Roadmap

- [ ] **Mobile Applications**: Native iOS and Android apps with offline capabilities
- [ ] **IoT Sensor Integration**: Real-time field monitoring through connected devices  
- [ ] **Drone Analytics**: Aerial crop surveillance and automated reporting
- [ ] **Blockchain Integration**: Supply chain transparency and traceability
- [ ] **Advanced Predictive Models**: Machine learning for yield optimization
- [ ] **Regional Language Expansion**: Support for local dialects and languages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Team Money Farmers**

For questions or feedback, please open an issue in this repository or reach out to the development team through GitHub.

---

<div align="center">
  <p><strong>Empowering Agriculture through Intelligent AI Solutions</strong></p>
  <p>KrishiMitra - Your AI-Powered Agricultural Companion</p>
</div>