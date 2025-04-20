# AI_Powered_Research_Paper_Analyzer

## Project Overview
The **AI-Powered Research Paper Analyzer** is a Python-based tool designed to streamline the analysis of academic research papers, with a focus on multilingual large language model (mLLM) evaluation. Built as a capstone project in a Kaggle notebook, it integrates **12 Generative AI capabilities**—including structured output, few-shot prompting, retrieval-augmented generation (RAG), embeddings, and MLOps monitoring—to extract insights, summarize content, and contextualize research papers. The tool processes a specific paper, *“Déjà Vu: Multilingual LLM Evaluation through the Lens of Machine Translation Evaluation”* by Julia Kreutzer et al., and connects it to a corpus of related works.

Key features include:
* **Structured Summaries:** Extracts title, authors, key findings, methodology, and implications in JSON format.
* **Concise Summaries:** Generates one-sentence summaries using few-shot prompting.
* **Contextual Analysis:** Uses RAG and vector search to retrieve relevant papers from a corpus.
* **Performance Monitoring:** Ensures high-quality outputs with a ROUGE-L F1 score ≥ 0.45 via MLOps.

This project demonstrates the power of Generative AI in making academic research accessible, efficient, and actionable for researchers, students, and professionals.

## Features
The analyzer leverages 12 AI capabilities to process research papers:
* **Structured Output:** Extracts key paper details in JSON format.
* **Few-Shot Prompting:** Produces concise, one-sentence summaries.
* **Document Understanding:** Provides detailed breakdowns with simplified explanations.
* **Agents:** Retrieves relevant papers based on queries.
* **Long Context Window:** Summarizes papers in-depth using large text inputs.
* **Context Caching:** Summarizes core arguments efficiently.
* **Gen AI Evaluation:** Evaluates summary quality using ROUGE metrics.
* **Grounding:** Contextualizes papers within existing research.
* **Embeddings:** Generates vector embeddings for semantic search.
* **Retrieval-Augmented Generation (RAG):** Answers queries using retrieved documents.
* **Vector Search:** Retrieves relevant papers via semantic similarity.
* **MLOps Monitoring:** Ensures performance with ROUGE-L F1 ≥ 0.45.

## Prerequisites
* **Kaggle Account:** Required to run the notebook.
* **Google Gemini API Key:** For AI model access.
* **Python Libraries:**
  * google-generativeai
  * chromadb
  * pypdf2
  * pandas
  * rouge-score

* **Input Files:**
  * Sample paper: Deja_Vu_Multilingual_LLM_Evaluation_through_the_Lens_of_Machine_Translation_Evaluation.pdf
  * Corpus: 10 open-access PDFs
  * Metadata: corpus_metadata.csv
 
## Setup Instructions
* **Create a Kaggle Notebook:**
  * Go to [Kaggle](https://www.kaggle.com/code) and create a new notebook.
  * Enable internet (Settings > Internet > On).
  * Set Accelerator to None (CPU).

* **Upload Input Files:**
  * **Sample Paper:** Upload `Deja_Vu_Multilingual_LLM_Evaluation_through_the_Lens_of_Machine_Translation_Evaluation.pdf` to `/kaggle/input/sample-paper/`.
  * **Corpus:** Upload 10 PDFs to `/kaggle/input/research-papers-corpus/`.
  * **Metadata:** Upload `corpus_metadata.csv` to `/kaggle/input/metadata/`.

* **Add Gemini API Key:**
  * Go to Notebook Settings > Add-ons > Secrets.
  * Add `GOOGLE_API_KEY` with your Gemini API key.

* **Install Dependencies:**
Run the following in a notebook cell:

    ```!pip install google-generativeai chromadb pypdf2 pandas rouge-score```

* **Run the Code:**
  * Copy the project code from the Kaggle Notebook (#) (or provided source).
  * Execute all cells to generate the 12 outputs.
 
## Project Structure

    ├── /kaggle/input/sample-paper/
    │   └── Deja_Vu_Multilingual_LLM_Evaluation_through_the_Lens_of_Machine_Translation_Evaluation.pdf
    ├── /kaggle/input/research-papers-corpus/
    │   ├── paper1.pdf
    │   ├── Unsupervised_Cross-lingual_Representation_Learning_at_Scale.pdf
    │   └── ...
    ├── /kaggle/input/metadata/
    │   └── corpus_metadata.csv
    ├── notebook.ipynb  # Main Kaggle notebook
    ├── README.md      # This file
    └── LICENSE        # MIT License

## Usage
* **Run the Notebook:**
  * Execute the notebook cells sequentially.
  * The tool processes the input paper and generates outputs for each of the 12 AI capabilities.

* **Key Outputs:**
  * **Structured Output:** JSON with paper details.
  * **Few-Shot Prompting:** One-sentence summary, e.g., “This paper advocates adopting machine translation evaluation methods to improve multilingual LLM evaluation.”
  * **Gen AI Evaluation:** ROUGE scores, e.g., ROUGE-L F1 = 0.4615.
  * **MLOps Monitoring:** Confirms performance with “Performance within acceptable limits (ROUGE-L F1 >= 0.45).”

* **Customize:**
  * Modify the input paper or corpus to analyze different research papers.
  * Adjust the `corpus_metadata.csv` to include relevant PDFs.
 
## Future Enhancements
* **Multi-Paper Analysis:** Process multiple papers in a single run.
* **Interactive UI:** Develop a web interface for user-friendly interaction.
* **Advanced Metrics:** Add BLEU or BERTScore for deeper evaluation.
* **Cloud Deployment:** Host on a cloud platform for scalability.

## Acknowledgments
* **Google Gemini:** For powering the AI capabilities.
* **Kaggle:** For providing a robust platform for notebook execution.
* **Open-Access Papers:** For enabling the corpus creation.

## Contact
For questions, feedback, or collaboration, reach out via [LinkedIn](https://www.linkedin.com/in/abhinandan-samal/) or [Email](samalabhinandan06@gmail.com).

*This project was developed as a capstone for [Gen AI Intensive Course Capstone 2025Q1/Google].*













