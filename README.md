# Inventory Forecasting Model 

This repository contains a **simple, practical inventory forecasting model** using Python and linear regression to predict weekly inventory demand, aiming to reduce overstock and understock scenarios in warehouse or retail environments.

## Features

✅ Uses historical sales data to forecast future demand.
✅ Visualizes predictions vs. actual data for interpretability.
✅ Saves the model for future integration into dashboards or WMS systems.
✅ Educational, beginner-friendly structure for learning ML in logistics.

## Project Structure

* `inventory_forecasting_model.py`: Main script for training, predicting, and visualizing.
* `inventory_data.csv`: Expected input CSV file containing `date`, `item_id`, and `units_sold` columns.
* `inventory_forecasting_model.pkl`: Saved model for re-use in dashboards.

## How It Works

1. Loads historical inventory sales data.
2. Aggregates data weekly for smoother trends.
3. Uses **scikit-learn Linear Regression** to predict next week's inventory needs.
4. Displays a plot comparing predicted vs. actual sales.
5. Prints evaluation metrics (MSE, R^2) for transparency.
6. Saves the trained model for real-time or batch deployment.

## Installation

1. Clone the repo:

```bash
git clone https://github.com/abedy101/Inventory-forecasting-model.git
cd Inventory-forecasting-model
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Replace `inventory_data.csv` with your dataset containing columns:

* `date` (YYYY-MM-DD)
* `item_id`
* `units_sold`

Then run:

```bash
python inventory_forecasting_model.py
```

## Future Improvements

✅ Add advanced models (Prophet, ARIMA, LSTM) for better seasonality handling.
✅ Automate SKU-level multi-item forecasting.
✅ Integrate with warehouse dashboards (e.g., Streamlit, Flask) for real-time visibility.
✅ Extend to reorder point and safety stock calculation.

## Author

**Abed**


---

**Feel free to fork and contribute to enhance this educational logistics ML project.**
