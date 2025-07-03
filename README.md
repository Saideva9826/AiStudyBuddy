# 🎓 AI Study Buddy – Context-Aware Q&A Chatbot for Students

**AI Study Buddy** is an intelligent Q&A system designed to assist students by answering course-related queries using advanced retrieval and generation techniques.

This project combines **RAG (Retrieval-Augmented Generation)** with **LangChain**, **OpenAI**, and **Streamlit** to create a responsive and context-aware assistant trained on custom learning materials.

---

## 🚀 Features

- 📚 Custom document ingestion with vector-based retrieval
- 🧠 Retrieval-Augmented Generation using OpenAI API
- 💬 Streamlit chatbot interface
- 🗂️ PDF/Text/Markdown document support
- 🔒 Secure local embedding with `FAISS` and LangChain

---

## 🛠️ Tech Stack

- Python, LangChain, OpenAI API
- Streamlit for UI
- FAISS for vector store
- Git, GitHub, VS Code

---



## 📁 Folder Structure

AiStudybuddy/
├── app/
│ ├── embedding_vectorstore.py
│ ├── rag_system.py
│ └── streamlit_app.py
├── static/
│ └── AI_Study_Buddy_Complete_Tutorial.pdf
├── templates/
│ └── index.html
├── notebooks/
│ ├── learning_guide.txt
│ ├── ssh_test.md
│ ├── todo.md
│ └── tutorial_phase2.md
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy
Edit

---

## 💻 How to Run It Locally

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Start the chatbot
streamlit run app/streamlit_app.py

🧪 Sample Use Cases
Ask: "What is vector similarity in LangChain?"

Ask: "Explain how this model works with PDFs?"

🙋‍♂️ Author

Saideva Goud Nathi
📫 LinkedIn • 🌐 GitHub

📝 License
This project is licensed under the MIT License.


---
https://choosealicense.com/licenses/mit/

---

