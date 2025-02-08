# Etymologistics (Work in Progress)

An open-source tool for visualizing and exploring etymological patterns using RAG (Retrieval Augmented Generation) and AI. This project combines traditional etymology resources with modern NLP techniques to provide interactive visualizations of word origins and evolution.

## 🚧 Project Status

This project is currently under active development. While core functionality is implemented, some features may be incomplete or subject to change. Contributions and feedback are welcome!

## 🌟 Features

### Core Functionality
- **Interactive Etymology Tree Visualization**: Explore word evolution through time with an interactive tree visualization
- **Morphological Analysis**: Break down words into their constituent stems, roots, and affixes
- **Multi-Source Research**: Combines data from:
  - Local etymonline dataset
  - Wiktionary API
  - Web scraping (etymonline.com)
  - AI-powered analysis

### Technical Features
- **RAG Pipeline**: Efficient retrieval and storage of etymology data using ChromaDB
- **Semantic Search**: Find similar words and patterns using embeddings
- **Caching System**: Local caching of research results for improved performance
- **Modular Architecture**: Clean separation of concerns between data sources, analysis, and visualization

## 🏗️ Architecture

The system is built with a modular architecture consisting of:

### Core Components
- `RAGPipeline`: Central coordinator for data storage and retrieval
- `EtymologyCache`: Persistent caching using ChromaDB
- `ResearchAgent`: Orchestrates etymology research across multiple sources
- `SimilarityAgent`: Handles word and stem similarity computations
- `TreeAgent`: Constructs hierarchical visualizations of word evolution
- `StemAgent`: Performs morphological analysis of words

### Data Sources
- Local etymonline dataset
- Wiktionary API integration
- Web scraping capabilities
- LLM-powered analysis

### Visualization
- Streamlit-based GUI with:
  - Interactive etymology trees
  - Timeline views
  - Stem analysis breakdowns
  - Similarity networks

## 🛠️ Setup

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/Etymologistics.git
cd Etymologistics
\`\`\`

2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Set up environment variables:
\`\`\`bash
# Create .env file with:
OPENAI_API_KEY=your_key_here  # Optional, for enhanced analysis
WIKTIONARY_API_KEY=your_key_here  # Optional
ETYMONLINE_API_KEY=your_key_here  # Optional
\`\`\`

4. Run the application:
\`\`\`bash
streamlit run src/gui.py
\`\`\`

## 📚 Usage

1. Enter a word in the search box
2. View the etymology tree visualization
3. Switch between different views:
   - Tree View: Interactive visualization of word evolution
   - Timeline View: Chronological representation of changes
   - Stem Analysis: Morphological breakdown of the word

## 🤝 Contributing

This project is open to contributions! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

[Add your chosen license here]

## 🙏 Acknowledgments

- Etymology data from etymonline.com
- Wiktionary API
- Sentence transformers for embeddings
- ChromaDB for vector storage
- Streamlit for visualization

## ⚠️ Limitations

- Some features require API keys for full functionality
- Web scraping is used as a fallback and may be unreliable
- LLM analysis quality depends on the model and prompt engineering
- Research results may vary in completeness and accuracy

## 🔜 Planned Features

- Enhanced visualization options
- Additional etymology data sources
- Improved similarity search
- Better handling of compound words
- Extended language family support
