import pandas as pd
import matplotlib.pyplot as plt
data = {
    'Cars': ['Benz', 'swift', 'rangerover', 'baleno'],
    'Number of cars': [25, 76, 45, 38],
    'Cost in lks': [45, 20, 75, 35]
}
df = pd.DataFrame(data)
# Bar Chart
plt.figure(figsize=(10, 5))
plt.bar(df['Cars'], df['Number of cars'], color='blue', label='cars')
plt.bar(df['Cars'], df['Cost in lks'], color='orange', label='Cost in lks', bottom=df['Number of cars'], alpha=0.7)
plt.xlabel('Cars')
plt.ylabel('costs')
plt.title('Bar Chart')
plt.legend()
plt.show()
# Line Chart
plt.figure(figsize=(10, 5))
plt.plot(df['Cars'], df['Number of cars'], marker='o', linestyle='-', color='blue', label='Number of cars')
plt.plot(df['Cars'], df['Cost in lks'], marker='o', linestyle='-', color='orange', label='Cost in lks')
plt.xlabel('Cars')
plt.ylabel('Costs')
plt.title('Line Chart')
plt.legend()
plt.show()
