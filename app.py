"""
Sri Lanka House Price Predictor.

Front end for the models trained in ML_assignment.ipynb.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent

MODEL_FAMILY = {
    "Random Forest": "Bagging",
    "XGBoost": "Boosting",
    "CatBoost": "Boosting",
    "Hybrid RF + CatBoost": "Hybrid",
    "Stacking": "Stacking",
}

PRICE_COL = "Price (LKR M)"
BASELINE = "Median Baseline"

st.set_page_config(page_title="Sri Lanka House Price Predictor", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(BASE / "models" / "stacking_model.pkl")


@st.cache_data
def load_metadata():
    with open(BASE / "models" / "metadata.json") as f:
        return json.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "data" / "clean_houses.csv", parse_dates=["posted_date"])
    df["price_million"] = df["price_lkr"] / 1_000_000
    return df


stack_model = load_model()
meta = load_metadata()
houses = load_data()
district_towns = meta["district_towns"]
metrics = pd.DataFrame(meta["metrics"])


def predict_all(row: pd.DataFrame) -> pd.DataFrame:
    """Price from every model, in LKR millions."""
    base = {name: float(est.predict(row)[0]) for name, est in stack_model.named_estimators_.items()}
    w = meta["blend_weight_rf"]
    out = {
        "Random Forest": base["Random Forest"],
        "XGBoost": base["XGBoost"],
        "CatBoost": base["CatBoost"],
        "Hybrid RF + CatBoost": w * base["Random Forest"] + (1 - w) * base["CatBoost"],
        "Stacking": float(stack_model.predict(row)[0]),
    }
    df = pd.DataFrame({"Model": list(out), PRICE_COL: list(out.values())})
    df.insert(1, "Type", df["Model"].map(MODEL_FAMILY))
    return df


st.title("Sri Lanka House Price Predictor")
st.write(
    f"Predicts the asking price of a house using ensemble models trained on {meta['n_train']:,} "
    f"ikman.lk ads ({meta['train_period'][0]} to {meta['train_period'][1]}). "
    f"Tested on the newest {meta['n_test']:,} ads."
)

tab_predict, tab_results, tab_data = st.tabs(["Predict", "Model results", "Data"])

# ---------------------------------------------------------------- predict
with tab_predict:
    districts = sorted(district_towns)
    c1, c2 = st.columns(2)
    district = c1.selectbox(
        "District", districts,
        index=districts.index("Colombo") if "Colombo" in districts else 0,
    )
    town = c2.selectbox("Town", district_towns[district])

    c1, c2 = st.columns(2)
    bedrooms = c1.number_input("Bedrooms", 1, 10, 3)
    bathrooms = c2.number_input("Bathrooms", 1, 10, 2)
    house_sqft = c1.number_input("House size (sqft)", 300, 20_000, 2_000, step=100)
    land_perches = c2.number_input("Land size (perches)", 1.0, 500.0, 10.0, step=0.5)

    row = pd.DataFrame([{
        "district": district,
        "location": town,
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "house_sqft": float(house_sqft),
        "land_perches": float(land_perches),
    }])
    preds = predict_all(row)
    final = preds.loc[preds["Model"] == "Stacking", PRICE_COL].iloc[0]
    stack_mae = metrics.loc[metrics["Model"] == "Stacking", "Test MAE"].iloc[0]
    town_prices = houses.loc[houses["location"] == town, "price_million"]

    st.divider()
    st.subheader("Estimated price")
    st.metric("Stacking model", f"Rs {final:,.1f} million")
    st.write(f"Typical error on unseen ads: about Rs {stack_mae:,.1f} million.")
    if len(town_prices) >= 5:
        st.write(
            f"For comparison, the median ad in {town} is Rs {town_prices.median():,.1f} million "
            f"({len(town_prices):,} listings)."
        )

    st.write("Prediction from each model")
    st.dataframe(
        preds.round(1), hide_index=True, width="stretch",
        column_config={PRICE_COL: st.column_config.NumberColumn(format="%.1f")},
    )

# ---------------------------------------------------------------- results
with tab_results:
    models_only = metrics[metrics["Model"] != BASELINE]
    best = models_only.iloc[0]

    st.write(
        f"Best model: **{best['Model']}** (test R² {best['Test R2']:.3f}, "
        f"test MAE Rs {best['Test MAE']:.2f} million)."
    )

    st.dataframe(
        metrics.drop(columns=["Family"]).rename(columns={
            "Train R2": "Train R²", "Test R2": "Test R²", "Gap": "Gap (R²)",
            "Train MAE": "Train MAE", "Test MAE": "Test MAE", "Test RMSE": "Test RMSE",
        }).round(3),
        hide_index=True, width="stretch",
    )
    st.caption(
        "MAE and RMSE are in LKR millions. Gap is train R² minus test R². A large gap suggests "
        "overfitting. Some of the gap is also due to prices changing over time, since the test "
        "ads are the newest."
    )

    fit_plot = models_only.melt(
        id_vars=["Model"], value_vars=["Train R2", "Test R2"], var_name="Set", value_name="R²"
    )
    fit_plot["Set"] = fit_plot["Set"].str.replace(" R2", "")
    fig = px.bar(
        fit_plot, x="Model", y="R²", color="Set", barmode="group",
        category_orders={"Model": list(models_only["Model"])},
        template="simple_white",
    )
    fig.update_layout(height=350, legend_title_text="", xaxis_title="", margin={"t": 20})
    fig.update_yaxes(range=[0.5, 1.0])
    st.plotly_chart(fig, width="stretch")

    st.write("Stacking weights given to each base model")
    weights = pd.DataFrame({
        "Model": list(meta["stack_weights"]),
        "Weight": [round(v, 2) for v in meta["stack_weights"].values()],
    })
    st.dataframe(weights, hide_index=True, width="stretch")

    st.subheader("Notes")
    st.markdown(
        """
- Random Forest (bagging) lowers variance but not bias, so it is the weakest ensemble here (test R² 0.730).
- Boosting does better because each tree corrects the previous ones. CatBoost (0.746) is slightly ahead of XGBoost (0.742).
- The hybrid of 0.35 Random Forest and 0.65 CatBoost scores best (test R² 0.750, MAE Rs 10.4 million).
- Stacking (0.749) is almost the same as the hybrid. Its meta-model puts most of the weight (0.78) on CatBoost.
- None of the models overfit badly. The train-test gap is only 0.07 to 0.08.
"""
    )

# ---------------------------------------------------------------- data
with tab_data:
    sel = st.multiselect("Districts (leave empty for all)", sorted(houses["district"].unique()))
    view = houses[houses["district"].isin(sel)] if sel else houses

    med = (view.groupby("district")["price_million"].agg(["median", "size"])
           .query("size >= 20").sort_values("median", ascending=False).reset_index())
    fig = px.bar(
        med, x="district", y="median", template="simple_white",
        labels={"median": "Median price (LKR M)", "district": ""},
    )
    fig.update_layout(title="Median asking price by district", height=350)
    st.plotly_chart(fig, width="stretch")

    sample = view.sample(min(3000, len(view)), random_state=42)
    fig = px.scatter(
        sample, x="house_sqft", y="price_million", opacity=0.4, template="simple_white",
        labels={"house_sqft": "House size (sqft)", "price_million": "Price (LKR M)"},
    )
    fig.update_layout(title="House size vs price (sample of listings)", height=350)
    fig.update_yaxes(range=[0, 250])
    fig.update_xaxes(range=[0, 8000])
    st.plotly_chart(fig, width="stretch")

with st.expander("About the data and models"):
    st.markdown(
        f"""
Data: house-for-sale ads from ikman.lk (public Kaggle dataset). After cleaning there are
{meta['n_train'] + meta['n_test']:,} listings. The target is the asking price in LKR millions, not the final sale price.

Split: by date. Trained on {meta['train_period'][0]} to {meta['train_period'][1]}, tested on
{meta['test_period'][0]} to {meta['test_period'][1]}.

Models: median baseline, Random Forest (bagging), XGBoost and CatBoost (boosting),
a Random Forest + CatBoost blend (weight {meta['blend_weight_rf']:.2f} on RF), and a stacking model
(Random Forest + XGBoost + CatBoost combined by a Ridge meta-model).
"""
    )

st.caption("Note: I used Claude (AI) for conceptual understanding and coding in this project.")
