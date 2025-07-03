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
            params={
                "DecisionTree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                },
                "RandomForest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                },
                "GradientBoosting": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.1],
                     'criterion': ['squared_error', 'friedman_mse']  # ✅ list is valid

                    },
                "Linear_regression": {},
                "K-Neighbors Classifier": {
                   'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    #'weights': ['uniform', 'distance'],
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                },
                "XGBClassifier": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.8, 0.9, 1.0], 
                },
                "CatBoosting classifier": {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [4, 6, 8],
                },
                "AdaBoost classifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                }

            }

            model_report: dict = evaluate_models(X_train=X_train, 
                                                 y_train=y_train,
                                                 X_test=X_test,
                                                 y_test=y_test,
                                                 models=models,
                                                 params=params)
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
