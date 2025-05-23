from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def tune_xgb_for_recall(X, y):
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1],
        'gamma': [0, 0.1, 0.2]
    }
    model = XGBClassifier(objective='binary:logistic', random_state=42)
    gs = GridSearchCV(
        model, 
        param_grid, 
        cv=3, 
        scoring='recall',  # MODIFIED: Оптимизация под Recall
        n_jobs=-1
    )
    gs.fit(X, y)
    return gs.best_params_
