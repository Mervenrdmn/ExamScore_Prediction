import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

with open("ExamScorePrediction.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    encoders = saved_data["encoders"]
    scaler = saved_data["scaler"]


class ExamScoreFeatures(BaseModel):
    age : int
    gender: str
    course: str
    study_hours: float
    class_attendance: float
    internet_access: str
    sleep_hours: float
    sleep_quality: str
    study_method: str
    facility_rating: str
    exam_difficulty: str




@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(features: ExamScoreFeatures):

    input_data = pd.DataFrame([features.model_dump()])

    input_data["internet_access"] = input_data["internet_access"].map({"yes": 1, "no": 2})
    input_data["gender"] = input_data["gender"].map({"male": 0, "female": 1})

    # Ordinal encoded features
    for col in ["sleep_quality", "facility_rating", "exam_difficulty"]:
        input_data[col] = encoders[col].transform(input_data[[col]])

    # OneHot encoded features
    course_encoded = encoders["course"].transform(input_data[["course"]])
    course_cols = [c.replace(" ", "_") for c in encoders["course"].get_feature_names_out(["course"])]
    input_data = input_data.drop(columns=["course"])
    input_data[course_cols] = course_encoded

    study_encoded = encoders["study_method"].transform(input_data[["study_method"]])
    study_cols = [c.replace(" ", "_") for c in encoders["study_method"].get_feature_names_out(["study_method"])]
    input_data = input_data.drop(columns=["study_method"])
    input_data[study_cols] = study_encoded

    # Numerical scaling
    numerical_columns = ["age", "study_hours", "class_attendance", "sleep_hours"]
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

    # column name compatibility
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    prediction = model.predict(input_data)

    return {"prediction": float(prediction[0])}