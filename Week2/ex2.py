Part 1

Download the AdSales.csv Download AdSales.csv from Canvas and move it to your week 2 exercises folder.

Take a look at the csv file and execute the following python code.

  
# import required libraries to your workspace

import pandas as pd

import matplotlib.pyplot as plt 

# read the csv file using pandas and copy the resulting dataframe to variable df

df = pd.read_csv('AdSales.csv')

# examine the dataframe df

print(df.head())

print(df.info())
print(df.shape)
print(df.columns)

What do you understand from the above results?

# draw a simple line plot of dataframe using the plot() function

df.plot()

plt.show()

Do you see any problems with this plot?

Change the python code to read the csv file in the beginning to

df = pd.read_csv('AdSales.csv', index_col='Quarter')

Re-execute all blocks and check if your line plot is fixed now. 

To save your plot as an image file adsales_lineplot.png, add another line of code.

plt.savefig('adsales_lineplot.png')

Check your workspace folder for saved image.

 

Part 2

Create a function sales_plot() and move the code you wrote so far into the function except the
code that imports pandas and matplotlib.

import ...
import ...
def sales_plot():

...

Create a new file python file and import your function in the new python file.

In the new file, call the function sales_plot()

Now execute the new python file in the terminal.

You just learnt how to create your own module and import it in your python code.

 

Part 3

Modify the sales_plot() function to take arguments csv_filename and index_column and pass
those arguments in your function call in the new python file.