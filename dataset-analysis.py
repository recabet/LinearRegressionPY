import matplotlib.pyplot as plt
import pandas as pd
import model
import numpy as np
import seaborn as sns
from scipy.stats import norm


df = pd.read_csv("c_class_listings.csv")

def to_azn (col):
	if "AZN" in col:
		return float(col.split()[0])
	return float(col.split()[0]) * 1.7


df2 = df[["Year", "Model", "Price", "Mielage", "Fuel type"]].copy()

df2["Price"] = df2["Price"].apply(to_azn)
df2.rename({"Price": "Price_AZN"}, axis="columns", inplace=True)

df2 = df2[df2["Fuel type"] != "Qaz"]
df2.rename({"Fuel type": "Fuel"}, axis="columns", inplace=True)


def remove_km (col):
	return int(col.strip()[:-2].replace(" ", ""))


df2["Mielage"] = df2["Mielage"].apply(remove_km)
df2.rename({"Mielage": "Mileage"}, axis="columns", inplace=True)

test_df = df2.drop("Model", axis=1)
test_df = pd.get_dummies(test_df)
test_df.drop("Fuel_Dizel", axis=1, inplace=True)
df2["Fuel"] = test_df["Fuel_Benzin"].copy()
df2 = df2[df2["Fuel"]]
df2.rename({"Fuel": "isBenzin"}, axis="columns", inplace=True)

sns.set(style="whitegrid")

fig, axs = plt.subplots(4, 1, figsize=(15, 15))

axs[0].scatter(df2["Mileage"], df2["Price_AZN"], alpha=0.6, edgecolors="r", s=80, color="r")
axs[0].set_title("Scatter Plot of Mileage vs Price", fontsize=16)
axs[0].set_xlabel("Mileage", fontsize=14)
axs[0].set_ylabel("Price (AZN)", fontsize=14)
axs[0].tick_params(axis="both", which="major", labelsize=12)
axs[0].grid(True, which='both', linestyle='--', linewidth=0.7)

axs[1].scatter(df2["Year"], df2["Price_AZN"], alpha=0.6, edgecolors="g", s=80, color="g")
axs[1].set_title("Scatter Plot of Price vs Year", fontsize=16)
axs[1].set_xlabel("Year", fontsize=14)
axs[1].set_ylabel("Price (AZN)", fontsize=14)
axs[1].tick_params(axis="both", which="major", labelsize=12)
axs[1].grid(True, which='both', linestyle='--', linewidth=0.7)

mileage = df2["Mileage"]
axs[2].hist(mileage, bins=50, alpha=0.75, color="skyblue", edgecolor="black", density=True)

mu, std = norm.fit(mileage)
xmin, xmax = axs[2].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
axs[2].plot(x, p, 'k', linewidth=2)

axs[2].set_title("Histogram of Mileage with Normal Distribution Curve", fontsize=16)
axs[2].set_xlabel("Mileage", fontsize=14)
axs[2].set_ylabel("Frequency", fontsize=14)
axs[2].tick_params(axis="both", which="major", labelsize=12)
axs[2].grid(True, which='both', linestyle='--', linewidth=0.7)

plt.show()

test_model = model.LinearRegression()
w, b = test_model.fit(np.array(df2[["Mileage", "Year"]]), np.array(df2["Price_AZN"]))

Mileage_range = np.linspace(df2["Mileage"].min(), df2["Mileage"].max(), 100)
Year_range = np.linspace(df2["Year"].min(), df2["Year"].max(), 100)
Mileage_mesh, Year_mesh = np.meshgrid(Mileage_range, Year_range)
Price_pred = w[0] * Mileage_mesh + w[1] * Year_mesh + b

fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection="3d")

ax.scatter(df2["Mileage"], df2["Year"], df2["Price_AZN"], c='b', marker='o')
ax.plot_surface(Mileage_mesh, Year_mesh, Price_pred, alpha=0.5, color='r')

ax.set_xlabel("Mileage")
ax.set_ylabel("Year")
ax.set_zlabel("Price")

plt.title("Linear Regression Model in 3D")
plt.tight_layout()
plt.show()
