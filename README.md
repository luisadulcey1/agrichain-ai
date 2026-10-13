
# AgriChain AI
### AI-Powered Complaint Resolution System

## Project Overview

AgriChain AI is an AI-powered complaint resolution system developed as a Final Capstone Project.

The system combines machine learning, natural language processing, retrieval-augmented generation (RAG), and multi-agent workflows to support the classification, investigation, and resolution of customer complaints.

The project also incorporates Human-in-the-Loop (HITL) review to ensure that important decisions and customer communications are validated by a human.

## Project Objectives

- Analyze customer complaint patterns and trends.
- Classify complaints using an Artificial Neural Network (ANN).
- Support human validation of predicted complaint categories.
- Retrieve relevant knowledge-base policies using FAISS.
- Coordinate specialized AI agents using LangGraph.
- Generate resolution plans and customer communication drafts.
- Require human approval before finalizing customer emails.
- Store complaint records and resolutions in SQLite.

## Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Artificial Neural Networks (ANN)
- Sentence Transformers / MiniLM
- LangGraph
- FAISS
- Retrieval-Augmented Generation (RAG)
- SQLite
- OpenAI API

## Application Dashboards

### Dashboard 1 — Exploratory Data Analysis

Provides visualizations of complaint distributions, trends, and other patterns identified during exploratory data analysis.

### Dashboard 2 — AI Complaint Classification

Uses an ANN-based classification workflow to predict complaint categories.

Includes human category validation, multi-agent workflow execution, customer email review, and SQLite case storage.

### Dashboard 3 — Multi-Agent Workflow Monitor

Demonstrates the LangGraph workflow and displays the outputs of specialized AI agents.

The workflow includes:

1. Analyzer
2. Escalation Manager
3. Investigator
4. Planner
5. Communicator

### Dashboard 4 — Resolution Management

Provides a persistent complaint management interface using SQLite.

Features include:

- Complaint registration
- Complaint status management
- Resolution plan management
- Customer email approval
- Complaint filtering and CSV export

## System Architecture

Customer Complaint
    |
    v
ANN Classification
    |
    v
Human Category Review
    |
    v
LangGraph Multi-Agent Workflow
    |
    +--> Policy Retrieval (FAISS / RAG)
    |
    v
Resolution Plan
    |
    v
Customer Email Draft
    |
    v
Human Email Approval
    |
    v
SQLite Case Storage
    |
    v
Resolution Management Dashboard

## Human-in-the-Loop

Human oversight is an essential component of AgriChain AI.

The system requires human validation of complaint categories before proceeding with the resolution workflow.

Customer communication drafts must also be reviewed and explicitly approved before being finalized.

The application does not automatically send customer emails.

## Data and Model Availability

The original datasets were provided for an academic capstone project and are not included in this public repository.

Locally generated model artifacts and certain processed datasets are also excluded.

Running the full application requires the appropriate datasets, trained model artifacts, knowledge-base files, and environment configuration.

## Project Status

Academic capstone prototype.

Four interactive Streamlit dashboards have been developed and tested locally, including integration with a persistent SQLite database.

## Author

Luisa Dulcey

Final Capstone Project — AI / Machine Learning
