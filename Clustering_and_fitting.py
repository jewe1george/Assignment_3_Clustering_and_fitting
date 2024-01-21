# Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

# Loading the dataset
data = pd.read_csv("marketing_campaign.csv", sep="\t")
print("Number of datapoints:", len(data))
data.head()

# Information on features
data.info()

# To remove the NA values
data = data.dropna()
print("The total number of data-points after removing the rows with missing values are:", len(data))

# Update the date format to "%d-%m-%Y"
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y")

# Extracting dates
dates = [date.date() for date in data["Dt_Customer"]]

# Dates of the newest and oldest recorded customer
print("The newest customer's enrollment date in the records:", max(dates))
print("The oldest customer's enrollment date in the records:", min(dates))

# Created a feature "Customer_For"
days = []
d1 = max(dates)  # taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

print("Total categories in the feature Marital_Status:\n",
      data["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n",
      data["Education"].value_counts())

# Feature Engineering
# Age of customer today
data["Age"] = 2021-data["Year_Birth"]

# Total spendings on various items
data["Spent"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + \
    data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]

# Deriving living situation by marital status"Alone"
data["Living_With"] = data["Marital_Status"].replace(
    {"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone", "Divorced": "Alone", "Single": "Alone", })

# Feature indicating total children living in the household
data["Children"] = data["Kidhome"]+data["Teenhome"]

# Feature for total members in the householde
data["Family_Size"] = data["Living_With"].replace(
    {"Alone": 1, "Partner": 2}) + data["Children"]

# Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children > 0, 1, 0)

# Segmenting education levels in three groups
data["Education"] = data["Education"].replace(
    {"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"})
# For clarity
data = data.rename(columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
                   "MntFishProducts": "Fish", "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})

# Dropping some of the redundant features
to_drop = ["Marital_Status", "Dt_Customer",
           "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
data = data.drop(to_drop, axis=1)

data.describe()

# To plot some selected features
# Setting up colors prefrences
sns.set(rc={"axes.facecolor": "#FFF9ED", "figure.facecolor": "#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(
    ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])


# Dropping the outliers by setting a cap on Age and income.
data = data[(data["Age"] < 90)]
data = data[(data["Income"] < 600000)]
print("The total number of data-points after removing the outliers are:", len(data))

# Select only numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[np.number])

# Create a correlation matrix
corrmat = numeric_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(20, 20))

# Draw the heatmap with seaborn
sns.heatmap(corrmat, annot=True, cmap='viridis', center=0)

# Show the plot
plt.show()

# Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)

# Label Encoding the object dtypes.
LE = LabelEncoder()
for i in object_cols:
    data[i] = data[[i]].apply(LE.fit_transform)

print("All features are now numerical")

# Creating a copy of data
ds = data.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
            'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)
# Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)
print("All features are now scaled")

# Scaled data to be used for reducing the dimensionality
print("Dataframe to be used for further modelling:")
scaled_ds.head()

# Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds),
                      columns=(["col1", "col2", "col3"]))
PCA_ds.describe().T

# Assuming you have a DataFrame 'PCA_ds' with columns "col1", "col2", "col3"

# Extracting data from DataFrame
x = PCA_ds["col1"]
y = PCA_ds["col2"]
z = PCA_ds["col3"]

# To plot with a constant color
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot with a constant color for all points
sc = ax.scatter(x, y, z, c='maroon', marker="o", alpha=0.8)

ax.set_title("Projection Of Data In The Reduced Dimension")


plt.show()

# Examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()

# Initiating the Agglomerative Clustering model
AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
# Adding the Clusters feature to the orignal dataframe.
data["Clusters"] = yhat_AC

# Plotting the clusters
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap=cmap)
ax.set_title("The Plot Of The Clusters")
plt.show()

# Set the style to whitegrid for better visibility
# Set a color palette
pal = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
sns.set(style="whitegrid")

# Plotting countplot of clusters
plt.figure(figsize=(8, 6))
pl = sns.countplot(x=data["Clusters"], palette=pal)

# Set title and labels
pl.set_title("Distribution Of The Clusters")
pl.set_xlabel("Clusters")
pl.set_ylabel("Count")

# Adding value annotations on top of the bars
for p in pl.patches:
    pl.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()

pl = sns.scatterplot(
    data=data, x=data["Spent"], y=data["Income"], hue=data["Clusters"], palette=pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()

plt.figure()
pl = sns.swarmplot(x=data["Clusters"], y=data["Spent"],
                   color="#CBEDDD", alpha=0.5)
pl = sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pal)
plt.show()

# Creating a feature to get a sum of accepted promotions
data["Total_Promos"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + \
    data["AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"]
# Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=data["Total_Promos"], hue=data["Clusters"], palette=pal)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()

# Plotting the number of deals purchased using boxenplot
pal = "Set2"
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
pl = sns.boxenplot(y=data["NumDealsPurchases"],
                   x=data["Clusters"], palette=pal)

# Set plot title and labels
pl.set_title("Number of Deals Purchased by Clusters")
pl.set_xlabel("Clusters")
pl.set_ylabel("Number of Deals Purchased")

# Show the plot
plt.show()

Personal = ["Kidhome", "Teenhome", "Customer_For", "Age",
            "Children", "Family_Size", "Is_Parent", "Education", "Living_With"]

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Flatten the axes for easy iteration
axes = axes.flatten()

# Loop over each personal feature and create a scatter plot with KDE
for i, feature in enumerate(Personal):
    sns.scatterplot(x=data[feature], y=data["Spent"],
                    hue=data["Clusters"], ax=axes[i], palette=pal, alpha=0.5)
    sns.kdeplot(x=data[feature], y=data["Spent"], fill=True,
                ax=axes[i], cmap="viridis", alpha=0.5)

    axes[i].set_title(f"{feature} vs Spent")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Spent")

# Remove empty subplots if any
for i in range(len(Personal), len(axes)):
    fig.delaxes(axes[i])

# Adjust layout for better spacing
plt.tight_layout()

# Add legend to the last subplot
axes[-1].legend(title="Clusters", loc="upper left", bbox_to_anchor=(1, 1))

# Show the plots
plt.show()
