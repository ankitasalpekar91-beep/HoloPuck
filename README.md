HoloPuck 🏒

AR Air Hockey using Python and Computer Vision

HoloPuck is a Python-based air hockey game that uses computer vision and hand tracking to create an interactive gaming experience. Instead of controlling the mallet using a mouse, the player can use their index fingertip in front of a webcam.

The project uses OpenCV to process live video and MediaPipe to detect hand landmarks. The detected fingertip position is used to control the virtual mallet, while the game logic handles puck movement, collisions, and scoring.

HoloPuck explores the practical use of image processing and computer vision in interactive applications, along with concepts related to Augmented Reality (AR), Virtual Reality (VR), and Mixed Reality (MR).

🎯 Project Objectives

- Develop an interactive air hockey game using Python.
- Use a webcam for real-time video input.
- Detect and track hand landmarks using MediaPipe.
- Control a virtual mallet through fingertip movements.
- Apply image processing and coordinate mapping techniques.
- Understand perspective transformation and virtual object placement.
- Explore the use of computer vision in AR-based applications.

✨ Features

- Real-Time Hand Tracking: Detects the player's hand through a webcam.
- Fingertip Detection: Uses the index fingertip as the control point for the virtual mallet.
- Virtual Air Hockey Table: Displays a game table with a puck and mallets.
- Interactive Gameplay: Allows the player to control the mallet using hand movements.
- Puck Physics: Handles puck movement, wall rebounds, and collisions.
- Scoring System: Tracks the scores during gameplay.
- Computer Opponent: Allows the player to compete against a computer-controlled opponent.
- Visual Feedback: Displays game objects and detected hand landmarks to help understand the tracking process.

Additional features, such as difficulty levels or alternative viewing modes, can be documented here when they are implemented and tested.

🛠️ Technologies Used

Technology| Purpose
Python| Main programming language and game logic
OpenCV| Webcam access, frame processing, and drawing game elements
MediaPipe Hands| Hand landmark detection and tracking
NumPy| Coordinate calculations and numerical operations
Webcam| Captures live video of the player's hand
GitHub| Stores and manages the project source code

🧠 Image Processing Concepts

1. Image Acquisition

The webcam captures live video frames that act as the input for the application.

2. Hand Landmark Detection

MediaPipe detects important points on the hand, including the index fingertip. These landmarks help the application understand hand position.

3. Real-Time Tracking

The detected fingertip position is updated across video frames so that the mallet can follow the player's movements.

4. Coordinate Mapping

The fingertip coordinates are used to determine the mallet's position within the game environment.

5. Perspective Transformation

A perspective transformation can map points between the camera view and a virtual table when the game uses a perspective-based layout.

6. Object Rendering

OpenCV draws the table, puck, mallets, and other visual elements onto the displayed frame.

7. Collision Detection

The game logic checks interactions between the puck, mallets, and table boundaries to update movement and gameplay.

⚙️ How It Works

The application follows these basic steps:

1. The webcam captures the player's hand in real time.
2. OpenCV reads the video frames.
3. MediaPipe detects the hand and its landmarks.
4. The index fingertip position is extracted.
5. The fingertip coordinates are mapped to the game table.
6. The virtual mallet follows the fingertip movement.
7. The puck moves according to the game physics and responds to collisions.
8. The score is updated when a goal is detected.
9. The updated game scene is displayed to the player.

📂 Project Structure

The project files can be organized in the following way:

HoloPuck/
│
├── README.md
├── LICENSE
├── .gitignore
├── main.py
└── requirements.txt

"main.py" represents the main application file, and "requirements.txt" lists the required Python libraries. The structure should be adjusted to match the actual files in the repository.

💻 Requirements

Hardware

- Laptop or desktop computer
- Working webcam
- Keyboard for supported game controls

Software

- Python 3.10–3.12, depending on library compatibility
- OpenCV
- MediaPipe
- NumPy

A dedicated VR headset, web frontend, backend server, or database is not required for the basic webcam-based game.

🚀 Getting Started

1. Clone the repository

git clone https://github.com/ankitasalpekar91-beep/HoloPuck.git

2. Open the project directory

cd HoloPuck

3. Install the dependencies

If a "requirements.txt" file is available:

pip install -r requirements.txt

Otherwise, install the main libraries:

pip install opencv-python mediapipe numpy

4. Run the application

python main.py

Use the actual Python filename if the main application has a different name. Make sure the required dependencies are installed before running the program.

🌐 Applications

The techniques explored in HoloPuck can also be useful in:

- Gesture-controlled games
- Touch-free computer interfaces
- Computer vision learning
- Interactive educational applications
- AR-based demonstrations
- Hand-tracking experiments

🔮 Future Scope

Some possible improvements for the project are:

- Improve hand-tracking accuracy and responsiveness.
- Add a two-player mode.
- Improve puck physics and collision handling.
- Introduce additional gameplay modes.
- Explore mobile AR implementation.
- Experiment with Unity or Vuforia for further AR development.

🏁 Conclusion

HoloPuck demonstrates how Python and computer vision can be combined to create an interactive air hockey game. By using a webcam, hand landmark detection, and real-time image processing, the project provides a practical way to explore gesture-based interaction and immersive technology concepts.

The main focus is on implementing the game and understanding the techniques involved, using accessible hardware and Python libraries.

---

Project Name: HoloPuck — AR Air Hockey
Domain: Image Processing and Computer Vision
Programming Language: Python
Tools: OpenCV, MediaPipe, NumPy
