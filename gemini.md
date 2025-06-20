# Wheatley 2.0: A Developer's Guide to Building a Personal AI Assistant

## 1. Executive Overview

Welcome to the development guide for Wheatley 2.0, a next-generation, privacy-first personal AI assistant. This document provides a comprehensive blueprint for a programming agent to understand the system's architecture, components, and the underlying philosophy driving its creation.

**Our Goal:** To build a highly personalized, proactive, and versatile AI assistant that lives on a local network, primarily interacting through a dedicated hardware device and a web-based portal. It will be a true digital companion, learning a user's habits, preferences, and context to provide timely and relevant assistance.

**Core Philosophy (Transitioning from Wheatley 1.0):** While Wheatley 1.0 laid the groundwork, Wheatley 2.0 represents a significant leap forward by focusing on a **single-user, deeply integrated experience**. This simplification eliminates the complexities of multi-user management, allowing for a more responsive, secure, and intimately personalized system. We are moving from a general-purpose tool to a dedicated personal agent.

This guide is structured to provide a clear path for implementation, covering the three main pillars of the project: the **Hardware Interface**, the **Backend Core**, and the **Web Portal**.

## 2. System Architecture

The Wheatley 2.0 ecosystem is designed for simplicity and efficiency, revolving around a central Raspberry Pi hub.

### 2.1. High-Level Diagram

```
+------------------------+      +---------------------+      +-----------------------+
|   Hardware Interface   |      |      Web Portal     |      |  External Services    |
| (Raspberry Pi Device)  |      |        (SPA)        |      |      (e.g., Google)   |
+------------------------+      +---------------------+      +-----------------------+
           |                              |                              |
           |   WebSocket (Secure)         |                              |
           |                              |                              |
           +--------------+---------------+------------------------------+
                          |
                          v
+------------------------------------------------------+
|             Backend Core (on Raspberry Pi)           |
|------------------------------------------------------|
|                  FastAPI Application                 |
|------------------------------------------------------|
| -> WebSocket Gateway                                 |
| -> Personal Agent (Gemini 2.5 Flash & MCP)           |
| -> Asynchronous Task Executor                        |
| -> Personal Memory System (SQLite & ChromaDB)        |
| -> Context Manager                                   |
| -> Voice Pipeline (STT/TTS)                          |
+------------------------------------------------------+
```

### 2.2. Technology Stack

* **Backend:** Python, FastAPI, Uvicorn
* **AI Model:** Google Gemini 2.5 Flash
* **Tool Integration:** MCP (Model-centric Communication Protocol)
* **Database:** SQLite (for structured data), ChromaDB (for vector embeddings)
* **Real-time Communication:** WebSockets
* **Hardware:** Raspberry Pi 4 (or newer), Microphone Array, Speaker, LEDs, Physical Buttons
* **Frontend:** Lightweight Single Page Application (SPA) - (e.g., using Vue.js, React, or Svelte)
* **Wake Word:** OpenWakeWord
* **Deployment:** Docker, Docker Compose

## 3. The Three Pillars of Wheatley 2.0

### 3.1. Pillar 1: The Hardware Interface (The "Alexa-style" Device)

This is the physical embodiment of Wheatley, designed to be an always-on, ambient assistant.

#### 3.1.1. Hardware Setup

* **Raspberry Pi:** The brain of the operation. A Raspberry Pi 4 with at least 4GB of RAM is recommended.
* **Microphone Array:** A multi-microphone array (e.g., ReSpeaker 4-Mic Array) is crucial for clear voice capture and direction-of-arrival detection.
* **Speaker:** A quality speaker for clear TTS responses.
* **LED Ring/Strip:** To provide visual feedback (e.g., listening, thinking, speaking).
* **Physical Buttons:** Connected to the Raspberry Pi's GPIO pins for quick actions like muting the microphone or triggering a specific routine.

#### 3.1.2. Software and Logic

A dedicated Python script will run on the Raspberry Pi with the following responsibilities:

* **Wake Word Detection:** Continuously listen for the "Hey Wheatley" wake word using a personalized model trained with OpenWakeWord. This should be highly efficient to run constantly in the background.
* **Audio Recording:** Upon wake word detection, record the user's command.
* **Communication with Backend:** Securely transmit the recorded audio to the backend via a WebSocket connection using a pre-configured API key.
* **Receiving Responses:** Listen for text responses from the backend over the WebSocket.
* **Text-to-Speech (TTS):** Convert the text response into speech and play it through the speaker.
* **LED Control:** Manage the LED patterns to reflect the assistant's state (e.g., blue for listening, pulsing white for thinking, green for speaking).
* **Button Input:** Monitor the GPIO pins for button presses and send corresponding commands to the backend.

### 3.2. Pillar 2: The Backend Core

This is the heart of Wheatley 2.0, running as a FastAPI application on the Raspberry Pi.

#### 3.2.1. FastAPI Application Structure

* **/ws:** The primary WebSocket endpoint for real-time communication with the hardware interface and web portal. Authentication will be handled via a simple API key passed as a query parameter.
* **/api/tasks:** RESTful endpoints for managing asynchronous tasks (e.g., checking the status).

#### 3.2.2. Core Components

* **Personal Agent (Gemini 2.5 Flash & MCP):** This component will be responsible for understanding user requests and orchestrating the necessary actions.
    * It will leverage Gemini 2.5 Flash for its powerful reasoning and language understanding capabilities.
    * **MCP (Model-centric Communication Protocol)** will be used for tool integration. This provides a standardized way for the agent to interact with various tools (e.g., web search, calendar, home automation). One of the initial tools will be a Gemini search with search grounding for reliable web-based answers.
* **Asynchronous Task Executor:** For tasks that may take time (e.g., in-depth research, generating a report), the system will immediately respond with "Sure, I'll get on that" and then execute the task in the background. Once completed, it will play a notification sound on the hardware device to get the user's attention. This allows for a non-blocking and responsive user experience. The system should be capable of handling multiple asynchronous tasks concurrently.
* **Personal Memory System:** A crucial component for making Wheatley feel intelligent and personalized.
    * **SQLite:** Will store structured memories, such as user preferences (e.g., "I am a vegetarian"), reminders, and scheduled actions. The schema will be simple and user-centric.
    * **ChromaDB:** Will store vector embeddings of conversational fragments and important information. This will allow for semantic search of past interactions, enabling Wheatley to recall contextually relevant information without needing explicit keywords. For example, if the user mentions a new project they are working on, this can be stored and later recalled when they ask for updates.
* **Context Manager:** This module will maintain the user's current context, including time of day, location (if available), and recent interactions. This information will be used to provide more proactive and relevant assistance.
* **Voice Pipeline:**
    * **Speech-to-Text (STT):** Transcribe the audio received from the hardware interface into text.
    * **Text-to-Speech (TTS):** Convert the agent's text responses into natural-sounding speech.

### 3.3. Pillar 3: The Web Portal

A lightweight, mobile-friendly Single Page Application (SPA) that provides a visual interface to Wheatley.

#### 3.3.1. Key Features

* **Secure Authentication:** The user will enter the same API key used by the hardware device to establish a secure WebSocket connection with the backend.
* **Real-time Interaction:** A chat-like interface for sending text-based commands to Wheatley and receiving responses in real-time.
* **Asynchronous Task Monitoring:** A section to view the status of ongoing and completed asynchronous tasks.
* **Memory Management (Optional):** A simple interface to view and manage some of Wheatley's stored memories and preferences.
* **Synchronized Experience:** Interactions through the web portal should be synchronized with the hardware device. For example, if a reminder is set via the web, the hardware device should deliver the notification at the appropriate time.

## 4. Key Features and Implementation Guidance

### 4.1. Asynchronous Work and User Feedback

When a user issues a complex command, the following flow should be implemented:

1.  The Personal Agent identifies the request as a long-running task.
2.  An immediate, low-latency response is sent back to the user (e.g., "I'm on it," "Working on that for you").
3.  The Asynchronous Task Executor starts the task in the background.
4.  The task's status is updated in the system (and visible on the web portal).
5.  Upon completion, the backend sends a specific "task complete" message over the WebSocket to the hardware device, which then plays a notification sound.

### 4.2. Memory and Personalization

* **Explicit Memory Storage:** When the user explicitly provides information (e.g., "My favorite color is blue," "Remind me to call my mom every Sunday"), this should be stored in the SQLite database with appropriate tags.
* **Implicit Memory Storage (Embeddings):** During conversations, the system should identify potentially important information and store embeddings of these conversational fragments in ChromaDB. For example, a discussion about an upcoming vacation should be stored for future reference.
* **Memory Retrieval:** Before processing a new request, the system should perform a semantic search on ChromaDB to retrieve relevant past interactions. These will be concisely summarized and included as context in the prompt to Gemini, allowing the assistant to "remember" past conversations.

### 4.3. Proactive Assistance

Leveraging the Context Manager, Wheatley should be able to offer proactive help. For example:

* **Morning Briefing:** Based on a user-defined time, Wheatley can provide a summary of the day's calendar events, weather, and important reminders.
* **Contextual Suggestions:** If the user is in the kitchen around dinnertime (inferred from time and past behavior), Wheatley might ask, "Would you like some recipe ideas for dinner?".

### 4.4. Security

The primary security measure will be a user-generated, secret API key. This key must be provided for any WebSocket connection to be accepted by the backend. All communication over the WebSocket should be encrypted (WSS).

### 4.5. Getting Creative: Advanced Features to Consider

* **Personal Knowledge Graph:** Evolve the memory system into a personal knowledge graph, connecting entities (people, places, projects) and their relationships. This will allow for more complex reasoning and a deeper understanding of the user's world.
* **Ambient Display Integration:** The hardware device could include a small display that shows contextual information (e.g., the current time, weather, upcoming appointments).
* **Multi-modal Input:** In the future, the web portal could be expanded to accept image and file uploads for the agent to process.
* **Personalized Routines:** Allow the user to create custom routines that chain multiple actions together (e.g., a "good morning" routine that turns on the lights, plays music, and starts the coffee maker).

## 5. Getting Started: A Phased Implementation Plan

This project can be broken down into manageable phases:

* **Phase 1: The Core Backend:**
    1.  Set up the FastAPI server with a secure WebSocket endpoint.
    2.  Integrate Gemini 2.5 Flash for basic question-answering.
    3.  Implement a basic STT/TTS pipeline.
* **Phase 2: The Hardware Interface:**
    1.  Set up the Raspberry Pi with the microphone and speaker.
    2.  Implement the wake word detection and audio streaming to the backend.
    3.  Enable TTS output on the hardware.
* **Phase 3: Memory and Asynchronicity:**
    1.  Set up the SQLite and ChromaDB databases.
    2.  Implement the memory storage and retrieval logic.
    3.  Build the asynchronous task execution framework.
* **Phase 4: The Web Portal:**
    1.  Develop the SPA with secure WebSocket communication.
    2.  Create the real-time chat interface.
    3.  Implement the task monitoring view.
* **Phase 5: Refinement and Advanced Features:**
    1.  Implement proactive assistance.
    2.  Integrate hardware buttons and LEDs.
    3.  Expand the MCP toolset.

This guide provides the foundational knowledge and direction for a programming agent to successfully build Wheatley 2.0. By focusing on a single-user, privacy-centric design, and leveraging powerful, modern AI tools, we can create a truly exceptional personal assistant. Let's get building!