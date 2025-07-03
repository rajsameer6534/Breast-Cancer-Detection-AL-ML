import os
import sys
from dataclasses import dataclass

from catboost import  CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.neighbors import  KNeighborsRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_models

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models= {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": GradientBoostingRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "Linear_regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting classifier": CatBoostRegressor(verbose=False),
                "AdaBoost classifier": AdaBoostRegressor(),
                
            }

            model_report: dict = evaluate_models(X_train=X_train, 
                                                 y_train=y_train,
                                                 X_test=X_test,
                                                 y_test=y_test,
                                                 models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
    
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")



            save_object(file_path=self.model_trainer_config.trained_model_file_path, 
                        obj=best_model)
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise CustomException(e, sys)
