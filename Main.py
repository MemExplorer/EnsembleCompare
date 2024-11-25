from flask import Flask, redirect, url_for, request
from flask import render_template
from EnsembleModels import AdaBoosting, BaggingClassicDecisionTree, BaggingRandomForest
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder="views", static_folder="static")

# information about each tested models
MODEL_INFO_DICT = {
    AdaBoosting: [
        "Ada Boosting",
        "Ada Boost, short for Adaptive Boosting, is a machine learning algorithm developed by Yoav Freund and Robert Schapire in 1995. It improves the performance of other learning algorithms by combining the outputs of multiple weak models into a weighted sum, creating a stronger final model. While primarily designed for binary classification, it can be extended to handle multiple classes or continuous values.",
    ],
    BaggingClassicDecisionTree: [
        "Bagging Classic Decision Tree",
        "A decision tree combined with bagging improves accuracy and reduces overfitting. Bagging creates multiple decision trees by training each on a random subset of the data. The final prediction is made by averaging the results (for regression) or taking a majority vote (for classification) from all the trees. This approach makes the model more reliable and stable.",
    ],
    BaggingRandomForest: [
        "Bagging Random Forest",
        "Bagging with Random Forest builds on the bagging technique by combining multiple decision trees, where each tree is trained on a random subset of the data and uses a random selection of features for splitting. The final prediction is made by averaging the results (for regression) or majority voting (for classification) from all the trees. This adds extra randomness, making the model more robust, accurate, and less prone to overfitting.",
    ],
}

# information about each clusters
CLUSTER_INFO = {
    "0": [
        "Low-Income Developing Country with High Child Mortality and Trade Dependence",
        """A low-income developing nation facing severe health challenges, moderate trade activity, and
                        economic instability due to high inflation and income disparities. The demographic and economic
                        indicators suggest a country with limited healthcare infrastructure, high fertility rates, and a
                        reliance on international trade.""",
    ],
    "1": [
        "Middle-Income Export-Oriented Country with Moderate Health and Trade Reliance",
        """A middle-income country with strong integration into global trade, moderate health outcomes, and
                        economic challenges such as trade dependence and inflation. The country demonstrates progress in
                        health and economic development but faces vulnerabilities linked to its reliance on imports and
                        exports.""",
    ],
    "2": [
        "High-Income Export-Driven Nation with Advanced Health and Trade Integration",
        """A high-income country with an advanced economy, exceptional health standards, and significant
                        integration into global trade. The country thrives on its export-oriented economy while
                        maintaining high living standards and a stable macroeconomic environment.""",
    ],
    "3": [
        "High-Income Export-Driven Nation with Strong Health and Moderate Trade Dependence",
        """A high-income country with robust trade activity, strong healthcare investment, and advanced
                        socio-economic indicators. The country benefits from a globally integrated economy and
                        demonstrates a high standard of living with exceptional health outcomes.""",
    ],
}

# uninitialized when init is not called
MODELS_BENCHMARK = []
MODEL_INSTANCE = None


def load_dataset():
    csv_path = "datasets/yeet.csv"
    csv_df = pd.read_csv(csv_path)
    df_values = csv_df.values
    no_countries = df_values[:, 1:]
    x_data = no_countries[:, :-1]
    y_data = np.unstack(no_countries[:, -1])
    return x_data, y_data


def init():
    x, y = load_dataset()
    labels = list(map(str, np.sort(np.unique(y))))
    benchmark_list = []
    model_inst_list = []
    for model_type in [AdaBoosting, BaggingClassicDecisionTree, BaggingRandomForest]:
        inst = model_type(seed=5)
        inst.fit(x, y)
        cf, accuracy, kappa = inst.benchmark_kfold(x, y)
        model_info = MODEL_INFO_DICT[model_type]

        benchmark_list.append(
            [model_info[0], model_info[1], accuracy, kappa, cf.tolist(), labels]
        )

        model_inst_list.append(inst)

    # select model with highest kappa value
    selected_model = max(zip(model_inst_list, benchmark_list), key=lambda x: x[1][3])[0]
    return benchmark_list, selected_model


def verify_keys(form, keys):
    return all(k in form for k in keys)


@app.route("/result", methods=["POST"])
def render_result():
    keys = [
        "under5Deaths",
        "exports",
        "healthSpending",
        "imports",
        "netIncome",
        "gdpGrowthRate",
        "lifeExpectancy",
        "fertilityRate",
        "gdpPerCapita",
    ]

    is_valid_input = verify_keys(
        request.form,
        keys,
    )

    if is_valid_input:
        str_input = list(map(request.form.get, keys))
        float_input = list(map(float, str_input))
        pred_result = MODEL_INSTANCE.predict_a([float_input])
        cluster_details = CLUSTER_INFO[str(pred_result[1])]
        return render_template(
            "result.jinja", pred_result=pred_result, cluster_details=cluster_details
        )


@app.route("/prediction")
def render_prediction():
    return render_template("prediction.jinja", cluster_details=CLUSTER_INFO)


@app.route("/")
def render_index():
    return render_template(
        "newindex.jinja", model_info=list(enumerate(MODELS_BENCHMARK))
    )


if __name__ == "__main__":

    # load each model and perform benchmark test
    print("Initializing models...")
    benchmark, model = init()
    MODELS_BENCHMARK = benchmark
    MODEL_INSTANCE = model

    # run web server
    print("Starting web server...")
    app.run(debug=True)
