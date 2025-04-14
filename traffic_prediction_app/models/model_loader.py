import joblib

class ModelHandler:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_df):
        return self.model.predict(input_df)
