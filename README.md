# Retail Store Video Analytics: Customer Behavior Tracking & Insights

**Author:** Karthik Raj S.

## Overview

This project implements an AI system designed to enhance the in-store customer experience in brick-and-mortar retail settings. It utilizes real-time video analytics from camera feeds to track customer movement, count specific objects (e.g., bottles), recognize customer actions, and analyze behavior patterns. The system logs processed data to a MongoDB database and provides insights through a Streamlit dashboard, addressing challenges faced by traditional retail in competing with personalized e-commerce experiences.

## Problem Statement Addressed

This project directly tackles **Problem Statement 4: Enhancing Customer Experience with AI-Driven Insights**.

<details><summary>Click to expand Problem Statement Details</summary>

**Objective:** Create an AI system that leverages real-time video analytics and customer behavior data to personalize in-store experiences.

**Prerequisites:**
*   Basic understanding of computer vision and video analytics.
*   Knowledge of AI/ML frameworks such as TensorFlow or PyTorch.
*   Experience with Python programming and handling large datasets.
*   Familiarity with real-time video processing tools like OpenCV.

**Problem Description:**
Brick-and-mortar retail stores face increasing competition from e-commerce platforms due to the lack of personalized experiences. The goal is to develop a smart AI system that:
1.  Tracks customer movement in real-time using in-store camera feeds.
2.  Analyzes customer behavior to identify high-traffic areas and customer preferences.
3.  Provides actionable insights for product placements, promotional strategies, and restocking shelves.

**Expected Outcomes:**
*   A functional prototype capable of tracking and analyzing real-time video feeds.
*   Real-time alerts for restocking or customer assistance in under-served areas (Insights provided via dashboard).
*   Insights for optimal product placement and promotion strategies based on behavior patterns.

**Challenges Involved:**
*   Processing multiple real-time video streams with minimal latency.
*   Ensuring accurate detection of customer behavior in varying lighting and crowded conditions.
*   Handling and analyzing large-scale data without overloading system resources.

**Tools & Resources Used (Examples):**
*   Hardware: Tested on standard PC, suitable for Intel AI PC with GPU/NPU for enhanced real-time performance.
*   Software: OpenCV, PyTorch (YOLOv8 & Custom Action Model), MongoDB, Streamlit, Pandas, Numpy, MediaPipe.
*   Datasets: Custom dataset for action recognition; relies on model detection capabilities for general objects/persons.

</details>

## Features

*   **Real-time Object Detection:** Uses YOLOv8 to detect persons and target objects (default: "bottle") in video streams.
*   **Multi-Person Tracking:** Employs a Centroid Tracker to assign and maintain unique IDs for detected persons across frames.
*   **Target Object Counting:** Counts the occurrences of the specified target object (e.g., bottles on a shelf) in each frame.
*   **Visit Analysis:** Calculates changes in target object count between a customer's entry and exit.
*   **Action Recognition (Optional):** Leverages MediaPipe for pose estimation and a custom PyTorch TSM-based model (`best_action_model.pth`) trained by Karthik Raj S. to classify actions like "standing", "walking", "reaching".
*   **Face Detection (Optional):** Includes an option to use a Caffe-based face detector within tracked person bounding boxes.
*   **Data Logging:** Records events (entry, exit, position updates, actions, bottle detections, camera config) with timestamps to a MongoDB database.
*   **Local Processing & Visualization (`realtime_feed.py`):**
    *   Processes video from file or webcam.
    *   Displays the video feed with bounding boxes, tracking IDs, detected actions, FPS, and bottle counts.
    *   Optionally overlays real-time heatmaps for person and bottle locations.
*   **Web Dashboard (`app.py`):**
    *   Connects to the MongoDB database.
    *   Visualizes historical data using Streamlit and Plotly.
    *   Displays metrics on bottle count changes, total visits, average change per visit.
    *   Shows traffic patterns (entries over time).
    *   Presents action distribution charts.
    *   Generates historical heatmaps for customer positions and bottle locations based on logged data.
    *   Allows filtering by date range and Camera ID.

## System Architecture


+--------------+ +-----------------------------------------+ +-----------+ +-----------------------------------------+ +----------------+
| Video Source | ----> | realtime_feed.py | ----> | MongoDB | <---- | app.py | ----> | User |
| (Webcam/File)| | (Detection, Tracking, Counting, Action) | | Database | | (Streamlit Dashboard, Data Fetch, Viz) | | (View Dashboard)|
| | | (Optional Display + Heatmaps) | | | | | | |
| | | (DB Logging) | | | | | | |
+--------------+ +-----------------------------------------+ +-----------+ +-----------------------------------------+ +----------------+

## Demo

*(You can add screenshots of `realtime_feed.py` running and the Streamlit dashboard here.)*

A snapshot of the Streamlit dashboard interface is included as `Retail AI Analytics Dashboard.mhtml`. To see the live dashboard, run `app.py`.

## Prerequisites

*   **Python:** 3.8 or higher recommended.
*   **MongoDB:** A running instance (local or remote). Get it from [https://www.mongodb.com/](https://www.mongodb.com/).
*   **Python Libraries:** See `requirements.txt`. Install using `pip install -r requirements.txt`.
*   **Hardware:** A standard PC. A GPU (NVIDIA with CUDA) is highly recommended for `realtime_feed.py` for real-time performance, especially with action recognition enabled.
*   **Model Files:**
    *   **YOLOv8:** Weights (`.pt` file like `yolov8n.pt`). Downloadable from Ultralytics or automatically downloaded on first run if needed.
    *   **Action Recognition Model:** Requires `best_action_model.pth` (provided by Karthik Raj S.). Place it where specified by the `--action_model_path` argument (or modify the default if needed).
    *   **Face Detector (Optional):** Requires `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`. Place them in a `models` directory or as specified by arguments.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create `requirements.txt` first using `pip freeze > requirements.txt` after installing packages manually if it's not included.)*

4.  **Set up MongoDB:** Ensure your MongoDB server is running.

5.  **Place Model Files:**
    *   Create a `models` directory if it doesn't exist.
    *   Place the optional face detector files (`deploy.prototxt`, `*.caffemodel`) inside `models/`.
    *   Place your custom `best_action_model.pth` file in a suitable location (e.g., `models/`) and note the path for the command-line argument.
    *   YOLOv8 models might be downloaded automatically by `ultralytics` or you can place a specific `.pt` file.

6.  **Configure Database Connection:**
    *   **Crucially:** The scripts connect to MongoDB using the `MONGO_URI` environment variable. If it's not set, they default to `mongodb://localhost:27017/`.
    *   For local default MongoDB (no auth): No action needed.
    *   For remote or authenticated MongoDB: **Set the `MONGO_URI` environment variable** before running the scripts.
        *   **Linux/macOS:**
            ```bash
            export MONGO_URI="mongodb://<user>:<password>@<host>:<port>/"
            ```
        *   **Windows (Command Prompt):**
            ```bash
            set MONGO_URI="mongodb://<user>:<password>@<host>:<port>/"
            ```
        *   **Windows (PowerShell):**
            ```bash
            $env:MONGO_URI="mongodb://<user>:<password>@<host>:<port>/"
            ```
    *   **Never hardcode sensitive connection strings directly in the code!**

## Running the Application

### 1. Local Processing (`realtime_feed.py`)

This script processes video input, performs analysis, optionally displays output, and logs data to MongoDB.

**Examples:**

*   **Run with webcam (index 0), display output, no action/face models:**
    ```bash
    python realtime_feed.py
    ```
*   **Run with webcam, display output, enable heatmaps:**
    ```bash
    python realtime_feed.py --show_heatmaps
    ```
*   **Run with video file, display output, enable action recognition:**
    ```bash
    python realtime_feed.py --input_video path/to/your/video.mp4 --action_model_path models/best_action_model.pth
    ```
*   **Run with video file, NO display (headless logging), YOLO-Nano, custom confidence:**
    ```bash
    python realtime_feed.py --input_video path/to/your/video.mp4 --no_display --yolo_model yolov8n.pt --yolo_conf 0.3
    ```
*   **Run with webcam, enable face detection (ensure models are in `models/`):**
    ```bash
    python realtime_feed.py --face_prototxt models/deploy.prototxt --face_weights models/res10_300x300_ssd_iter_140000.caffemodel
    ```

Use `python realtime_feed.py --help` to see all available options. Press 'q' in the display window to quit.

### 2. Streamlit Dashboard (`app.py`)

This script runs a web server to display the analytics dashboard based on data in MongoDB.

**Run:**
```bash
streamlit run app.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Navigate to the local URL provided by Streamlit in your web browser. Ensure realtime_feed.py has run previously to populate the database with some data.

Custom Action Recognition Model

The optional action recognition feature uses a custom model (best_action_model.pth) developed by the author, Karthik Raj S.

Architecture: Based on Temporal Shift Modules (TSM), defined in realtime_feed.py (EnhancedTSM, ActionRecognitionModel).

Training Data: Trained on a custom dataset of ~150 videos capturing relevant actions (standing, walking, reaching).

Status: This model is part of ongoing research aimed at publication in a conference or journal.

To use it, provide the path to best_action_model.pth via the --action_model_path argument when running realtime_feed.py.

Future Work / Potential Improvements

Implement real-time alert mechanisms based on insights (e.g., low bottle count, specific actions detected).

Scale the system for multiple simultaneous camera feeds.

Deploy to cloud infrastructure for wider accessibility.

Enhance dashboard with more sophisticated analytics and filtering options.

Improve model accuracy and robustness (ongoing).

Integrate with store inventory or POS systems.

License

(Choose a license, e.g., MIT, Apache 2.0, and add the corresponding LICENSE file to your repository)

Example: This project is licensed under the MIT License - see the LICENSE file for details.

Contact & Citation

Karthik Raj S. - (Optionally add contact info like LinkedIn profile or email)

If you use this code or the custom action model in your research, please consider citing this repository and acknowledging the author's work, especially concerning the action recognition model currently under research for publication.

**Before Committing:**

1.  **Create `requirements.txt`:** If you haven't already, activate your virtual environment and run:
    ```bash
    pip freeze > requirements.txt
    ```
2.  **Add a LICENSE file:** Choose an open-source license (like MIT or Apache 2.0), create a file named `LICENSE`, and paste the license text into it. Update the "License" section in the README accordingly.
3.  **Add Screenshots:** Replace the placeholder text with actual screenshots of your application running.
4.  **Verify Paths:** Double-check that the default paths mentioned (like for face models in `models/`) match your intended structure or update the README/code arguments.
5.  **Review:** Read through the generated README one last time to ensure accuracy and clarity.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
