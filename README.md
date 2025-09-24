# Animal-Type-Classification-for-cattle-and-buffaloes
Here’s a complete `README.md` template for your Flask-based Cattle/Buffalo classification project, including setup and production deployment instructions.

***

# Animal-Type-Classification-for-Cattle-and-Buffaloes

This web application classifies uploaded cattle or buffalo images into their specific breeds and animal type (cow/buffalo) using a pre-trained TensorFlow Keras model. It is built using Flask and serves an easy-to-use UI for image upload and prediction.

***

## Features

- **Upload any cattle or buffalo image**
- **Get breed and animal type (cow/buffalo) prediction**
- **See prediction confidence percentage**
- **Simple web frontend**
- **Easy local setup, with production deployment instructions**

***

## Directory Structure

```
├── app.py
├── labels.csv
├── models/
│   └── fine_tuned_model_mobilenet_v2_fine_tuned.keras
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
├── templates/
│   ├── index.html
│   └── result.html
├── requirements.txt
├── README.md
└── .gitignore
```

***

## Setup Instructions (Development)

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your-username>/Animal-Type-Classification-for-cattle-and-buffaloes.git
   cd Animal-Type-Classification-for-cattle-and-buffaloes
   ```

2. **Create & Activate a Virtual Environment**

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Prepare Model & Labels**

   - Put the trained Keras model file in `models/fine_tuned_model_mobilenet_v2_fine_tuned.keras`
   - Make sure class names `labels.csv` is in the root directory.

5. **Run the Application (Dev Server)**

   ```bash
   python app.py
   ```
   The web app starts at [http://127.0.0.1:5000](http://127.0.0.1:5000)

***

## Usage

- Visit the web interface.
- Upload an image of cattle or buffalo.
- See the predicted breed, type, and confidence shown with the image.

***

## Requirements

See `requirements.txt` for full list. Main packages:
- flask
- tensorflow
- keras
- pandas
- numpy
- werkzeug
