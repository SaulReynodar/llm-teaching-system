# LLM Teaching System with MiRo

<p align="center">
  <img src="media/gifs/full_demo.gif" width="700"/>
</p>

---

## Overview

This project presents an AI-based interactive teaching system that integrates a Large Language Model (LLM) with the MiRo robot to simulate a classroom environment.

The system enables real-time communication between a human user, an AI teacher, and the MiRo robot. It combines speech processing, intelligent dialogue generation, and robotic behaviour to create a natural and engaging learning experience.

---

## Aim of the System

The aim of this project is to develop an interactive AI-driven classroom system using the MiRo robot.

### Objectives

- Enable real-time communication between human, robot, and AI  
- Use LLMs for intelligent conversation (Teacher and MiRo)  
- Integrate speech technologies (Speech-to-Text and Text-to-Speech)  
- Create a natural and engaging learning experience  

---

## System Demonstration

### Process Flow

1. Human speech is captured using a microphone  
2. Speech is converted to text using Whisper (Speech-to-Text)  
3. The Teacher LLM processes the input  
   - Understands context (story or question)  
   - Generates an appropriate response  
4. MiRo robot interacts as a student  
   - Asks a question during the story  
   - Answers teacher questions  
5. The response is converted to speech using OpenAI TTS (GPT-4o-mini-tts)  
6. Communication between modules is handled through ROS topics  
7. The system ensures real-time interaction with smooth turn-taking  

---

## Demonstration (GIFs)

### MiRo Listening Behaviour
<p align="center">
  <img src="media/gifs/miro_nodding.gif" width="600"/>
</p>

### Teacher Storytelling
<p align="center">
  <img src="media/gifs/teacher_story.gif" width="600"/>
</p>

### MiRo Asking a Question
<p align="center">
  <img src="media/gifs/miro_question.gif" width="600"/>
</p>

### Reaction to Correct Answer
<p align="center">
  <img src="media/gifs/miro_spin.gif" width="600"/>
</p>

---

## System Architecture

<p align="center">
  <img src="media/images/system_architecture.png" width="700"/>
</p>

### Core Components

- Speech-to-Text (Whisper)  
  Converts human speech into text  

- LLM (AI Teacher – Shared Model)  
  Processes inputs from both human and MiRo  
  Generates context-aware responses  

- Text-to-Speech (OpenAI GPT-4o-mini-TTS)  
  Converts generated text into speech  

- MiRo Robot  
  Executes behaviours and gestures based on responses  

- ROS Topics  
  Enables real-time communication between all modules  

---

## Latency and Design Choices

### System Pipeline

- Speech input → Speech-to-Text (Whisper)  
- Text → LLM processing  
- Response → Text-to-Speech  
- Output → Audio playback  

### Design Decisions

- OpenAI GPT-4o-mini-TTS is used for the teacher voice to provide a natural and realistic output  
- Piper TTS was not used for the teacher due to lower voice quality  
- A child-like voice is used for MiRo  

### Performance

- Total response time: approximately 3–6 seconds  
- Despite slight delay, interaction remains smooth and natural  

---

## MiRo Behaviour

The following behaviours were implemented:

- Dynamic ear movement during listening  
- Responding to teacher questions  
- Reaction to correct answers using motion  
- Reaction to incorrect answers through posture change  

---

## Human Interaction and Evaluation

### Interaction

- Users can ask questions during the session  
- The system processes and responds using AI  
- Interaction follows a structured and controlled flow  

### Turn-Taking Mechanism

- Only one agent speaks at a time  
- The teacher pauses when the human speaks  
- Responses are generated only after input is complete  
- Prevents overlapping speech  

### Evaluation Criteria

- Ease of interaction  
- Clarity of responses  
- Natural conversation flow  
- Behaviour of the MiRo robot  
- Overall user experience  

### Purpose

- Assess user perception of the system  
- Measure interaction quality  
- Identify areas for improvement  

---

## System Implementation

The system consists of:

- ROS nodes for Teacher, MiRo, and Speech modules  
- Real-time communication using ROS topics  
- Integration of Speech-to-Text, LLM, and Text-to-Speech pipeline  

---

## Technologies Used

- ROS Noetic  
- Gazebo 11  
- Python  
- OpenAI GPT-4o-mini  
- OpenAI Text-to-Speech  
- Whisper (Speech-to-Text)  

---

## Project Structure
