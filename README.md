# Hatem - Academic Assistant

## Overview

Hatem is an intelligent academic assistant designed to provide a range of functionalities, specifically tailored to the **University of Hail's policies and regulations**. These include:

- **Smart Chat (Built with RAG)**: An interactive assistant utilizing Retrieval-Augmented Generation (RAG) to provide accurate and context-aware responses.
- **Attendance Calculator**: A tool to track and calculate attendance records.
- **GPA Calculator**: A quick way to compute GPA based on entered grades.
- **Useful Educational Resources**: A collection of tools and materials to enhance productivity and learning.

## Project Structure

The repository is organized as follows:

- **Hatem-frontend/**: The frontend application built using React.
- **Hatem-backend/**: The backend powered by FastAPI, including the RAG implementation.
  - **rag_api_compatible_v3_2_openai.py**: A Python file implementing retrieval-augmented generation (RAG).

## Features

- 🗣️ **AI-Powered Chat (RAG)**: Engage in natural language interactions with enhanced retrieval-based responses, focusing on University of Hail's academic policies and regulations.
- 📊 **Attendance Management**: Easily track attendance.
- 🎓 **GPA Calculation**: Quickly determine GPA scores.
- 📚 **Educational Tools**: Access a variety of useful materials.

## Installation

### Environment Variables

A `.env` file should be placed inside the `Hatem-backend/` directory to store the OpenAI API key. The file should contain:

```ini
OPENAI_API_KEY=your_openai_api_key_here
```

To run the project locally, follow these steps:

### Backend Setup (FastAPI)

1. Navigate to the backend directory:
   ```bash
   cd Hatem-backend
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI server (which includes the RAG implementation):
   ```bash
   uvicorn app:app --reload
   ```

### Frontend Setup (React)

1. Navigate to the frontend directory:
   ```bash
   cd Hatem-frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the React development server:
   ```bash
   npm run dev
   ```

## Usage

Once both the backend and frontend are running, open the frontend in your browser and interact with Hatem.

## Contribution

If you'd like to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and commit them.
4. Submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For inquiries or support, please contact **faisal.yalthuwaini@gmail.com** or open an issue in the repository.

---

Feel free to update this README as the project evolves!
