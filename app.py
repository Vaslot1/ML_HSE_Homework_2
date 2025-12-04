import marimo

__generated_with = "0.18.1"
app = marimo.App()

with app.setup:
    # Initialization code that runs before all other cells
    pass


@app.cell
def _(mo):
    mo.md("""
    # 💻 Computer Price Estimator Dashboard
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    import matplotlib.pyplot as plt
    import seaborn as sns
    return KFold, XGBRegressor, cross_val_score, mo, np, pd, plt, sns


@app.cell
def _():
    USD_TO_RUB_RATE = 77.0
    return (USD_TO_RUB_RATE,)


@app.cell
def _(mo):
    mo.md("""
    ### 1. Data Loading & Training
    """)
    return


@app.cell
def _(pd):
    # Load Data (Ensure 'computer_prices_all.csv' is in the same folder)
    data = pd.read_csv("computer_prices_all.csv")
    return (data,)


@app.cell
def _(data, pd):
    # --- Preprocessing Logic ---
    # Drop any duplicate columns to prevent errors
    cleaned_data = data.loc[:, ~data.columns.duplicated()]

    # 1. Target Encoding Mappings
    brand_means = cleaned_data.groupby('brand')['price'].mean()
    form_factor_means = cleaned_data.groupby('form_factor')['price'].mean()

    # 2. Ordinal Mappings
    resolution_order = {'1920x1080': 1, '2560x1440': 2, '2560x1600': 3,
                        '2880x1800': 4, '3440x1440': 5, '3840x2160': 6}
    display_order = {'LED': 1, 'IPS': 2, 'QLED': 3, 'Mini-LED': 4, 'OLED': 5, 'VA': 6}

    # 3. Apply Mappings
    data_processed = cleaned_data.copy()
    data_processed['brand_encoded'] = data_processed['brand'].map(brand_means)
    data_processed['form_factor_encoded'] = data_processed['form_factor'].map(form_factor_means)
    data_processed['resolution_encoded'] = data_processed['resolution'].map(resolution_order).fillna(1)
    data_processed['display_type_encoded'] = data_processed['display_type'].map(display_order).fillna(2)

    # 4. One-Hot Encoding
    features_onehot = ['os', 'cpu_brand', 'gpu_brand', 'device_type']
    data_encoded = pd.get_dummies(data_processed, columns=features_onehot, prefix=features_onehot)

    # Align columns for training
    numerical_features = ['gpu_tier', 'cpu_tier', 'ram_gb', 'cpu_cores',
                          'cpu_threads', 'cpu_base_ghz', 'cpu_boost_ghz', 'vram_gb']
    encoded_cols = [col for col in data_encoded.columns if 'encoded' in col or any(f in col for f in features_onehot)]

    # Final Feature Set
    final_features = encoded_cols + numerical_features
    # Ensure uniqueness while preserving order
    final_features = list(dict.fromkeys(final_features))

    # Filter to ensure we only use numeric columns
    feature_columns = [col for col in final_features if col in data_encoded.columns]

    X = data_encoded[feature_columns]
    y = cleaned_data["price"]
    return (
        X,
        brand_means,
        display_order,
        feature_columns,
        form_factor_means,
        resolution_order,
        y,
    )


@app.cell
def _(KFold, USD_TO_RUB_RATE, X, XGBRegressor, cross_val_score, mo, np, y):
    # Perform Cross-Validation
    model_for_cv = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Using neg_mean_absolute_error as the scoring metric
    scores = cross_val_score(model_for_cv, X, y, cv=kfold, scoring='neg_mean_absolute_error')

    # Convert scores to positive and calculate mean and std
    mae_scores_usd = -scores
    mean_mae_usd = np.mean(mae_scores_usd)
    std_mae_usd = np.std(mae_scores_usd)

    mean_mae_rub = mean_mae_usd * USD_TO_RUB_RATE
    std_mae_rub = std_mae_usd * USD_TO_RUB_RATE

    mo.md(f"""
    ### ⚙️ Model Validation (5-Fold CV)
    To ensure the model is reliable, we use 5-fold cross-validation. 
    The model can predict the price with an average error of **{mean_mae_rub:,.2f} RUB** (or ${mean_mae_usd:,.2f}).
    *(Standard Deviation of error: {std_mae_rub:,.2f} RUB)*
    """)
    return


@app.cell
def _(X, XGBRegressor, y):
    # Train Final Model on All Data
    # This model is used for the interactive predictions below
    model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return (model,)


@app.cell
def _(
    brand_means,
    data,
    display_order,
    form_factor_means,
    mo,
    resolution_order,
):
    mo.md("### 🎛️ Configure Computer Specs")

    # Get unique values for dropdowns
    brands = sorted(list(brand_means.index))
    os_list = sorted(data['os'].unique().tolist())
    form_factors = sorted(list(form_factor_means.index))
    resolutions = sorted(list(resolution_order.keys()), key=lambda x: resolution_order[x])
    display_types = sorted(list(display_order.keys()), key=lambda x: display_order[x])
    cpu_brands = sorted(data['cpu_brand'].unique().tolist())
    gpu_brands = sorted(data['gpu_brand'].unique().tolist())
    device_types = sorted(data['device_type'].unique().tolist())

    # Create all UI elements
    brand_dropdown = mo.ui.dropdown(options=brands, label="Brand")
    form_factor_dropdown = mo.ui.dropdown(options=form_factors, label="Form Factor")
    os_dropdown = mo.ui.dropdown(options=os_list, label="OS")
    device_type_dropdown = mo.ui.dropdown(options=device_types, label="Device Type")

    cpu_brand_dropdown = mo.ui.dropdown(options=cpu_brands, label="CPU Brand")
    gpu_brand_dropdown = mo.ui.dropdown(options=gpu_brands, label="GPU Brand")

    resolution_dropdown = mo.ui.dropdown(options=resolutions, label="Resolution")
    display_type_dropdown = mo.ui.dropdown(options=display_types, label="Display Type")

    ram_slider = mo.ui.slider(8, 128, 8, value=16, label="RAM (GB)")
    vram_slider = mo.ui.slider(2, 24, 2, value=8, label="VRAM (GB)")

    gpu_tier_slider = mo.ui.slider(1, 6, 1, value=3, label="GPU Tier")
    cpu_tier_slider = mo.ui.slider(1, 6, 1, value=3, label="CPU Tier")

    cpu_cores_slider = mo.ui.slider(2, 32, 2, value=8, label="CPU Cores")
    cpu_threads_slider = mo.ui.slider(4, 64, 4, value=16, label="CPU Threads")

    cpu_base_ghz_slider = mo.ui.slider(start=2.0, stop=4.0, step=0.1, value=2.6, label="CPU Base (GHz)")
    cpu_boost_ghz_slider = mo.ui.slider(start=2.8, stop=5.0, step=0.1, value=3.5, label="CPU Boost (GHz)")

    # Layout

    # Grouping controls for a better dashboard layout
    group_general = mo.vstack([
        brand_dropdown,
        os_dropdown,
        form_factor_dropdown,
        device_type_dropdown
    ])

    group_cpu = mo.vstack([
        cpu_brand_dropdown,
        cpu_tier_slider,
        cpu_cores_slider,
        cpu_threads_slider,
        cpu_base_ghz_slider,
        cpu_boost_ghz_slider
    ])

    group_gpu_ram = mo.vstack([
        gpu_brand_dropdown,
        gpu_tier_slider,
        ram_slider,
        vram_slider,
    ])

    group_display = mo.vstack([
        resolution_dropdown,
        display_type_dropdown,
    ])

    # Create an accordion for each group and stack them horizontally
    dashboard = mo.hstack([
        mo.accordion({"Основные": group_general}),
        mo.accordion({"Процессор": group_cpu}),
        mo.accordion({"Графика и память": group_gpu_ram}),
        mo.accordion({"Дисплей": group_display})
    ])

    mo.callout(dashboard)

    # Return all controls
    return (
        brand_dropdown,
        cpu_base_ghz_slider,
        cpu_boost_ghz_slider,
        cpu_brand_dropdown,
        cpu_cores_slider,
        cpu_threads_slider,
        cpu_tier_slider,
        device_type_dropdown,
        display_type_dropdown,
        form_factor_dropdown,
        gpu_brand_dropdown,
        gpu_tier_slider,
        os_dropdown,
        ram_slider,
        resolution_dropdown,
        vram_slider,
    )


@app.cell
def _(
    USD_TO_RUB_RATE,
    brand_dropdown,
    brand_means,
    cpu_base_ghz_slider,
    cpu_boost_ghz_slider,
    cpu_brand_dropdown,
    cpu_cores_slider,
    cpu_threads_slider,
    cpu_tier_slider,
    device_type_dropdown,
    display_order,
    display_type_dropdown,
    feature_columns,
    form_factor_dropdown,
    form_factor_means,
    gpu_brand_dropdown,
    gpu_tier_slider,
    model,
    os_dropdown,
    pd,
    ram_slider,
    resolution_dropdown,
    resolution_order,
    vram_slider,
):
    # --- Prediction Logic ---

    # 1. Create a dictionary from the UI controls
    input_data = {
        'brand': [brand_dropdown.value],
        'form_factor': [form_factor_dropdown.value],
        'os': [os_dropdown.value],
        'device_type': [device_type_dropdown.value],
        'cpu_brand': [cpu_brand_dropdown.value],
        'gpu_brand': [gpu_brand_dropdown.value],
        'resolution': [resolution_dropdown.value],
        'display_type': [display_type_dropdown.value],
        'ram_gb': [ram_slider.value],
        'vram_gb': [vram_slider.value],
        'gpu_tier': [gpu_tier_slider.value],
        'cpu_tier': [cpu_tier_slider.value],
        'cpu_cores': [cpu_cores_slider.value],
        'cpu_threads': [cpu_threads_slider.value],
        'cpu_base_ghz': [cpu_base_ghz_slider.value],
        'cpu_boost_ghz': [cpu_boost_ghz_slider.value],
    }
    df_input = pd.DataFrame(input_data)

    # 2. Apply all encodings on the fly
    # Target Encoding
    df_input['brand_encoded'] = df_input['brand'].map(brand_means)
    df_input['form_factor_encoded'] = df_input['form_factor'].map(form_factor_means)
    # Ordinal Encoding
    df_input['resolution_encoded'] = df_input['resolution'].map(resolution_order)
    df_input['display_type_encoded'] = df_input['display_type'].map(display_order)

    # 3. Align to Training Columns and fill known values
    df_ready = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in df_input.columns:
        if col in df_ready.columns:
            df_ready[col] = df_input[col].values
        # Also copy the encoded columns we just made
        encoded_col_name = f"{col}_encoded"
        if encoded_col_name in df_input.columns and encoded_col_name in df_ready.columns:
            df_ready[encoded_col_name] = df_input[encoded_col_name].values

    # 4. Manually apply One-Hot Encoding for prediction
    one_hot_features = {
        'os': os_dropdown.value,
        'cpu_brand': cpu_brand_dropdown.value,
        'gpu_brand': gpu_brand_dropdown.value,
        'device_type': device_type_dropdown.value
    }
    for feature, value in one_hot_features.items():
        col_name = f"{feature}_{value}"
        if col_name in df_ready.columns:
            df_ready[col_name] = 1

    # 5. Predict in USD and convert to RUB
    prediction_usd = model.predict(df_ready[feature_columns].values)[0]
    prediction = prediction_usd * USD_TO_RUB_RATE
    return (prediction,)


@app.cell
def _(mo, prediction):
    mo.md(f"""
    ## 💰 Estimated Price: **{prediction:,.2f} RUB**
    """)
    return


@app.cell
def _(mo, prediction):
    budget = 350000
    if prediction <= budget:
        budget_message = f"🎉 **Отлично!** Сборка укладывается в ваш бюджет в **{budget:,.0f} RUB**."
        budget_kind = "success"
    else:
        over_by = prediction - budget
        budget_message = f"🤔 **Превышение бюджета.** Эта сборка на **{over_by:,.0f} RUB** дороже вашего бюджета."
        budget_kind = "danger"

    mo.callout(budget_message, kind=budget_kind)
    return (budget,)


@app.cell
def _(feature_columns, model, pd):
    # Create a DataFrame for feature importances, to be used by other cells
    importances = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    return (importances,)


@app.cell
def _(importances, mo, plt, sns):
    mo.md("### 📊 What Drives the Price?")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Fix for seaborn FutureWarning: assign 'y' to 'hue' and hide legend
    sns.barplot(
        data=importances,
        x='importance',
        y='feature',
        hue='feature',
        palette='viridis',
        ax=ax,
        legend=False
    )

    ax.set_title('Top 10 Most Important Features for Price Prediction')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()

    # The 'fig' object is the last expression, so Marimo will display it.
    fig
    return


@app.cell
def _(budget, importances, mo, prediction):
    mo.md("### 💡 Recommendations")

    # Get the most important feature
    top_feature = importances['feature'].iloc[0]

    # Simple recommendation logic
    if prediction > budget:
        rec_message = f"Чтобы снизить стоимость, попробуйте уменьшить параметр с наибольшим влиянием на цену: **{top_feature}**."
        rec_kind = "warn"
    elif budget - prediction < 20000: # If very close to budget
        rec_message = "Вы почти у цели! Конфигурация отлично сбалансирована по цене."
        rec_kind = "info"
    else:
        rec_message = f"У вас есть запас в бюджете. Вы можете улучшить самый влиятельный параметр: **{top_feature}**, чтобы получить больше производительности."
        rec_kind = "success"

    mo.callout(rec_message, kind=rec_kind)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
