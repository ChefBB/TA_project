### PRELIMINARY DATA EXPLORATION AND VISUALIZATION

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


path = '.\\models\\data\\'
df = pd.read_csv(path + 'lab_lem_merge.csv')
df



plt.figure(figsize= (10,6))
df.groupby('label').count()['id'].sort_values(ascending= False).plot(kind= 'bar')
plt.xlabel('Class')
plt.ylabel('Number of rows')
plt.xticks(rotation= 0.9)
plt.title('Number of occurrences for each class')
plt.grid(axis= 'y', alpha= 0.3)
plt.show()

plt.close()

"""
Visualize the distribution of class occurrences in the dataset containing around 100'000 stanzas.

This script uses Matplotlib to create a bar chart that shows the number of rows 
for each class in a dataset. The data is grouped by the `label` column, and the 
counts are plotted in descending order.

Steps:
1. A figure is created with a specified size (10x6 inches).
2. The dataset is grouped by the `label` column, and the counts of the `id` column 
   are calculated.
3. The counts are sorted in descending order and plotted as a bar chart.
4. Labels for the x-axis and y-axis, as well as a title for the chart, are added.
5. The y-axis grid is enabled with a specified transparency.
6. The plot is displayed and the figure is closed.

"""


data = df.groupby('label').count()['id'].sort_values(ascending= False)

percentages = (data / data.sum()) * 100
labels = [f"{label} ({value:.0f}%)" for label, value in zip(data.index, percentages)]

cmap = get_cmap('prism')
colors = [cmap(i / len(data)) for i in range(len(data))]

# Plot the pie chart
plt.figure(figsize=(10, 7))
plt.pie(
    data, 
    labels=labels,  # Use the custom labels
    colors= colors,
    startangle=90   # Rotate the chart to start at 90 degrees
)

# Add a title
plt.title('Distribution of Categories')
plt.show()

plt.show()

plt.close()

"""
Generate a pie chart to visualize the distribution of categories in the dataset.

This script creates a pie chart to show the relative proportions of each category 
in a dataset, with percentage labels for each slice. Custom colors are used for 
the slices based on a colormap.

Steps:
1. Data is grouped by the `label` column, and the counts of the `id` column are 
   calculated and sorted in descending order.
2. Percentages for each category are calculated based on the total count.
3. Custom labels are generated for each category, including the percentage values.
4. A colormap (`prism`) is used to assign unique colors to each slice.
5. A pie chart is created with:
   - Custom labels for each slice.
   - Colors assigned from the colormap.
   - A starting angle of 90 degrees for rotation.
6. The chart is displayed with a title.

"""
