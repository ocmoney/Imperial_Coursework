import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)


def summarize_data(df, show_head=10):
    """Print basic summaries: shape, columns, head, dtypes and describe()."""
    print("Data shape:", df.shape)
    print("\nColumn names:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head(show_head))
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics (describe):")
    # include a few percentiles for income
    print(df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))


def analyze_income_and_cars(df):
    """Print income and car ownership distributions and category counts."""
    print("\nIncome distribution (value counts for income bins/indexes):")
    # If income is recorded as discrete categories (integers) this prints counts per value
    print(df['Annual income'].value_counts().sort_index().head(50))

    print("\nCar ownership distribution:")
    print(df['Cars'].value_counts().sort_index())

    print("\nIncome statistics:")
    print("Min income:", df['Annual income'].min())
    print("Max income:", df['Annual income'].max())
    print("Income percentiles:")
    print(df['Annual income'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

    # Category counts used in the model
    low_income_count = (df['Annual income'] <= 20).sum()
    med_income_count = ((df['Annual income'] > 20) & (df['Annual income'] <= 70)).sum()
    high_income_count = (df['Annual income'] > 70).sum()

    print(f"\nIncome categories:")
    print(f"Low income (<=20k): {low_income_count} households")
    print(f"Medium income (21k-70k): {med_income_count} households")
    print(f"High income (>70k): {high_income_count} households")
    print(f"Total: {low_income_count + med_income_count + high_income_count}")

    # Car ownership categories
    zero_cars = (df['Cars'] == 0).sum()
    one_car = (df['Cars'] == 1).sum()
    two_plus_cars = (df['Cars'] >= 2).sum()

    print(f"\nCar ownership:")
    print(f"0 cars: {zero_cars} households")
    print(f"1 car: {one_car} households")
    print(f"2+ cars: {two_plus_cars} households")
    print(f"Total: {zero_cars + one_car + two_plus_cars}")


def quick_plots(df):
    """Create a couple of quick exploratory plots (optional)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Histogram of Annual income
    axes[0].hist(df['Annual income'], bins=30, edgecolor='black')
    axes[0].set_title('Annual income distribution')
    axes[0].set_xlabel('Annual income (k)')
    axes[0].set_ylabel('Count')

    # Bar of car ownership counts
    car_counts = df['Cars'].value_counts().sort_index()
    axes[1].bar(car_counts.index.astype(str), car_counts.values, edgecolor='black')
    axes[1].set_title('Car ownership distribution')
    axes[1].set_xlabel('Number of cars')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_file = "Data_CW1.csv"
    print("Loading data from:", data_file)
    data = load_data(data_file)

    print("\n--- Summary ---")
    summarize_data(data)

    print("\n--- Income and Cars Analysis ---")
    analyze_income_and_cars(data)

    # Optional quick plots - uncomment if you want to see them
    quick_plots(data)
