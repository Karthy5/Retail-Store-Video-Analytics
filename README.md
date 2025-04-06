# AI-Driven Retail Analytics System (Intel Industrial Training Project)

**Author:** Karthik Raj S.

## Project Overview

This project is submitted as part of the **Intel Industrial Training Program**, addressing **Problem Statement 4: Enhancing Customer Experience with AI-Driven Insights**.

The goal is to create an AI system that leverages real-time video analytics from in-store cameras to understand customer behavior and provide actionable business insights. The system tracks customer movement, analyzes actions, monitors product interaction (specifically bottle counts in this implementation), and presents findings through an interactive dashboard.

**Key Innovation:** A core component of this project is **JATE (Joint-Aware-Temporal-Encoder) , a custom Action Recognition model (`jate_model.pth`)**, developed entirely by the author. This involved:
*   Single-handedly creating a **custom dataset** comprising 150 videos relevant to retail scenarios.
*   Designing and training a **unique neural network architecture** (using PyTorch) specifically for recognizing actions like "reaching," "standing," and "walking" within the store environment.
*   *Currently authoring a research paper on this model and dataset for potential publication in a conference or journal.*
*   Link to JATE: https://github.com/Karthy5/JATE-Joint-Aware-Temporal-Encoder-

This system demonstrates a practical application of computer vision and machine learning to solve real-world retail challenges, aligning with the objectives outlined in the Intel program.

## Core Features

*   **Real-time Object Detection:** Utilizes YOLOv8 to detect `person` and `bottle` objects in video streams.
*   **Multi-Person Tracking:** Implements a Centroid Tracker to assign unique IDs and track individuals across frames.
*   **Custom Action Recognition:** Employs the bespoke `jate_model.pth` (powered by MediaPipe pose estimation) to classify customer actions ("reaching", "standing", "walking").
*   **Product Interaction Monitoring:** Tracks the count of detected bottles, calculating changes associated with customer visits (entry/exit).
*   **Comprehensive Event Logging:** Records key events (entry, exit, position updates, actions, bottle detections, configuration) to MongoDB for persistent storage and analysis.
*   **Interactive Dashboard:** A Streamlit application (`app.py`) provides visualization of:
    *   Bottle count changes and visit analysis.
    *   Store traffic patterns over time.
    *   Distribution of observed customer actions.
    *   Historical heatmaps for customer movement (person positions).
    *   Historical heatmaps for product locations (bottle positions).
*   **Local Processing & Visualization:** The core script (`realtime_feed.py`) can run locally, displaying the processed video feed with overlays (bounding boxes, IDs, actions) and optional real-time heatmaps.
*   **Configuration Management:** Uses environment variables (`MONGO_URI`) for secure and flexible database connection.

## Technology Stack

*   **Programming Language:** Python 3.x
*   **Computer Vision:** OpenCV, MediaPipe
*   **Object Detection:** Ultralytics YOLOv8
*   **Deep Learning:** PyTorch (for the custom action model)
*   **Database:** MongoDB
*   **Dashboard:** Streamlit
*   **Data Handling:** Pandas, NumPy
*   **Visualization:** Plotly
*   **Hardware:** Designed to leverage Intel Hardware (CPU/GPU/NPU where applicable, though tested on standard configurations).

## Setup Instructions

1.  **Prerequisites:**
    *   Python 3.8+ and Pip
    *   Git
    *   MongoDB instance (local or remote/cloud)

2.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

3.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download/Place Models:**
    *   **YOLOv8:** The script defaults to `yolov8n.pt` or `yolov8s.pt`. If these are not automatically downloaded by `ultralytics`, download the desired weights file (e.g., from [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)) and place it in the project root or provide the path via argument.
    *   **Face Detector (Optional):** The script uses default paths (`models/deploy.prototxt`, `models/res10_300x300_ssd_iter_140000.caffemodel`). If you wish to use face detection, create a `models` directory and place these Caffe model files inside. You can often find these online (search for the filenames).
    *   **Custom Action Model:** Place your pre-trained `jate_model.pth` file in the project root or specify its path using the `--action_model_path` argument when running `realtime_feed.py`. **This model is crucial for the action recognition feature.**

6.  **Configure MongoDB:**
    *   **Default:** The scripts default to connecting to `mongodb://localhost:27017/`. If you have a local MongoDB running on the default port without authentication, no further action is needed.
    *   **Remote/Authenticated:** For connecting to a different host, port, or an authenticated database (e.g., MongoDB Atlas), **set the `MONGO_URI` environment variable** before running the scripts.
        ```bash
        # Example on Linux/macOS
        export MONGO_URI="mongodb+srv://<username>:<password>@<your-cluster-url>/<db-name>?retryWrites=true&w=majority"
        # Example on Windows (Command Prompt)
        set MONGO_URI="mongodb+srv://<username>:<password>@<your-cluster-url>/<db-name>?retryWrites=true&w=majority"
        # Example on Windows (PowerShell)
        $env:MONGO_URI="mongodb+srv://<username>:<password>@<your-cluster-url>/<db-name>?retryWrites=true&w=majority"
        ```
        Replace the placeholders with your actual credentials and cluster details. Both `realtime_feed.py` and `app.py` will use this environment variable.

## Usage

Ensure your MongoDB instance is running and accessible, and the `MONGO_URI` is set correctly if needed.

1.  **Run Local Real-time Analysis (`realtime_feed.py`):**
    *   This script processes a video source, performs detection/tracking/action recognition, logs data to MongoDB, and optionally displays the output.
    *   **From Webcam (Index 0):**
        ```bash
        python realtime_feed.py --action_model_path jate_model.pth [--show_heatmaps]
        ```
    *   **From Video File:**
        ```bash
        python realtime_feed.py --input_video path/to/your/video.mp4 --action_model_path jate_model.pth [--show_heatmaps] [--no_display]
        ```
    *   **Key Arguments:**
        *   `--input_video`: Path to video file (uses webcam if omitted).
        *   `--cam_index`: Webcam index (default 0).
        *   `--yolo_model`: Path to YOLO model (default `yolov8n.pt`).
        *   `--action_model_path`: **Required** Path to your custom action model (`jate_model.pth`).
        *   `--face_prototxt`, `--face_weights`: Paths for optional face detection.
        *   `--no_display`: Run without showing the video window (headless analysis/logging).
        *   `--show_heatmaps`: Overlay real-time heatmaps on the displayed video (requires display).
        *   `--device`: Set processing device (`auto`, `cpu`, `cuda`).

2.  **Launch the Analytics Dashboard (`app.py`):**
    *   This script runs a web server to display the Streamlit dashboard using data from MongoDB.
    *   **Run the command:**
        ```bash
        streamlit run app.py
        ```
    *   Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
    *   Use the sidebar controls to filter data by date range and Camera ID.

## Project Structure


.
├── realtime_feed.py # Main script for real-time video processing and logging
├── app.py # Streamlit dashboard application
├── jate_model.pth # Custom action recognition model
├── models/ # Optional: Directory for face detector models
│ ├── deploy.prototxt
│ └── res10_300x300_ssd_iter_140000.caffemodel
├── Retail AI Analytics Dashboard.mhtml # Static snapshot of the dashboard UI
├── requirements.txt # Python dependencies
└── README.md # This file

## Dashboard Demonstration

A static representation of the Streamlit dashboard UI is included as `Retail AI Analytics Dashboard.mhtml`. This file can be opened in a web browser to get a visual impression of the dashboard's layout and features. For the live, interactive experience, please run `app.py`.

![alt text](https://github.com/Karthy5/Retail-Store-Video-Analytics/blob/main/Screenshots/Screenshot%202025-04-05%20230711.png?raw=true)
![alt text](https://github.com/Karthy5/Retail-Store-Video-Analytics/blob/main/Screenshots/Screenshot%202025-04-05%20230723.png?raw=true)
![alt text](https://github.com/Karthy5/Retail-Store-Video-Analytics/blob/main/Screenshots/Screenshot%202025-04-05%20230755.png?raw=true)
![alt text](https://github.com/Karthy5/Retail-Store-Video-Analytics/blob/main/Screenshots/Screenshot%202025-04-05%20230824.png?raw=true)

## Future Work & Potential Enhancements

*   **Advanced Action Recognition:** Expand the custom model to recognize more nuanced behaviors (e.g., product examination, dwell time near specific shelves).
*   **Real-time Alerting:** Implement triggers based on insights (e.g., low bottle count alert, potential assistance needed based on prolonged dwell time).
*   **POS Integration:** Correlate visit data and bottle changes with actual sales data for deeper insights.
*   **Performance Optimization:** Further optimize processing pipelines, potentially leveraging Intel NPU/GPU capabilities more explicitly.
*   **Scalability:** Architect for handling feeds from numerous cameras simultaneously (e.g., using message queues, distributed processing).
*   **Cloud Deployment:** Package the system for deployment on cloud platforms.

## Acknowledgments

This project was developed as part of the Intel Industrial Training Program. I appreciate the opportunity provided by Intel to work on this challenging and relevant problem statement.
Use code with caution.
