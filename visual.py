import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/training_label.csv", nrows=840)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data into a DataFrame
df = pd.read_csv('../data/training_label.csv')  # Replace 'your_data.csv' with the actual file path


# Comparing Urban rural dependency for Skilled birth attendant rate : 
y_param = 'Stunted_Rate'

# Create a box plot
sns.boxplot(data=df, x='URBAN_RURA', y=y_param)
plt.title("Plot")
plt.xlabel('URBAN_RURA')
plt.ylabel(y_param)
plt.show()

