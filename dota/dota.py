import pandas

data = pandas.read_csv('features.csv')
# for i in (0, data.columns.size):
#     print('Columns is ', data.columns[i])
#     print('Value count is', data.columns[i].value_counts)
#     print('--------------------------------------------')
print(data.isnull().sum())
# with open("NaN.txt", "w") as output:
#     output.write(str(data.isnull().sum()))
