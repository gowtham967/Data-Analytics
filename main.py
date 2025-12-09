from data_preprocessing import load_and_clean_data
from exploratory_analysis import plot_attrition_distribution, correlation_heatmap
from model_training import train_models

# Step 1: Load and clean data
df = load_and_clean_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Step 2: Exploratory Analysis
plot_attrition_distribution(df)
correlation_heatmap(df)

# Step 3: Train Models
train_models(df)
