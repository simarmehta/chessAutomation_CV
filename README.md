# Chess Automation Using Computer Vision

This repository showcases a project that leverages **Computer Vision (CV)** techniques to automate chessboard detection and piece tracking in real-time. The project allows users to track a physical chess game through a camera, identifying the pieces and their positions on the board, and determining the game moves.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project uses **OpenCV** and **Numpy** to process images or video feeds of a chessboard. It detects the board, identifies the pieces, and tracks the game's progress by recognizing piece movements.

## Features

- **Chessboard Detection**: Automatically recognizes a chessboard from a camera feed.
- **Piece Recognition**: Detects both black and white pieces on the board.
- **Move Detection**: Tracks the movement of pieces to log game history.
- **Real-time Processing**: Operates in real-time, ideal for live games.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- OpenCV (`opencv-python`)
- Numpy (`numpy`)

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/simarmehta/chessAutomation_CV.git
    cd chessAutomation_CV
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

   *(If `requirements.txt` is missing, install dependencies manually: `pip install opencv-python numpy`)*

## Usage

1. Ensure your camera is connected or use a pre-recorded video feed.
2. Run the main script:

    ```bash
    python chess_automation.py
    ```

3. The script will start detecting the chessboard and pieces and display the board's state in real-time.

## How It Works

1. **Board Detection**: The chessboard is detected using contour detection techniques. The perspective is adjusted to ensure the board is correctly aligned.
2. **Piece Recognition**: After segmenting the chessboard into 64 squares, the algorithm identifies pieces based on their shape and color properties.
3. **Move Tracking**: By comparing the current frame with previous ones, the system detects any changes in piece positions, allowing it to log the moves.

### Techniques Used:

- **Edge Detection** for finding the chessboard boundaries.
- **Perspective Warping** to straighten the chessboard view.
- **Thresholding** and **Image Processing** for recognizing chess pieces.

## Project Structure

```bash
chessAutomation_CV/
├── chess_automation.py   # Main script to run the automation
├── utils/                # Helper functions (image processing, etc.)
├── data/                 # Sample data (optional)
├── models/               # Models or pre-trained weights (if applicable)
└── README.md             # Documentation
