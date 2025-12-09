import matplotlib.pyplot as plt
import seaborn as sns
def plot_attrition_distribution(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='Attrition', data=df)
    plt.title("Attrition Distribution")
    plt.show()
def correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.show()
