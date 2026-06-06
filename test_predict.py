import pandas as pd
import pickle

with open("calorie_predictor.pkl", "rb") as f:
    model = pickle.load(f)

sample = pd.DataFrame({
    "Gender": ["male"],
    "Age": [25],
    "Height": [170],
    "Weight": [70],
    "Duration": [30],
    "Heart_Rate": [110],
    "Body_Temp": [37.0]
})

print(model.predict(sample))