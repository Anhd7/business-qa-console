import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_all_models_and_rank(df, topic_col="Business Head"):
    topic_col = topic_col.lower()  

    models = {}
    best_model_map = {}

    for idx, row in df.iterrows():
        topic = str(row[topic_col]).strip().lower()

        values = []
        for q in ['q1', 'q2', 'q3', 'q4']:
            try:
                val = float(row[q])
                if pd.notna(val):
                    values.append(val)
            except:
                pass

        if len(values) < 2:
            continue

        X = [[i + 1] for i in range(len(values))]
        y = values

        lr = LinearRegression()
        rf = RandomForestRegressor()

        lr.fit(X, y)
        rf.fit(X, y)

        y_pred_lr = lr.predict(X)
        y_pred_rf = rf.predict(X)

        mse_lr = mean_squared_error(y, y_pred_lr)
        mse_rf = mean_squared_error(y, y_pred_rf)

        avg_growth = lambda x: [y[-1] + ((y[-1] - y[0]) / (len(y) - 1)) * (x[0] - len(y))]

        model_set = {
            "linear_regression": lr,
            "random_forest": rf,
            "average_growth": avg_growth
        }

        best_model = min(
            [("linear_regression", mse_lr), ("random_forest", mse_rf)],
            key=lambda x: x[1]
        )[0]

        models[topic] = model_set
        best_model_map[topic] = best_model

    return models, best_model_map
