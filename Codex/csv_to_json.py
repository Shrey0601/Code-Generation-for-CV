import csv
import json
from numpy import NaN
import pandas as pd


# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
	
	# create a dictionary
	df = pd.read_csv(csvFilePath, encoding= 'utf-8')

	key = []
	print(df.size)

	n = 53

	for i in range(n):
		val = {}
		val['nl'] = df['NL'][i]
		val['code'] = df['PL'][i]
		key.append(val)

	with open("Classification.json", "w") as final:
		for d in key:
			if(d['code']!="NaN"):
				json.dump(d, final)
				final.write('\n')
	

	
		
# Driver Code

# Decide the two file paths according to your
# computer system
csvFilePath = r"Classification dataset - Sheet1.csv"
jsonFilePath = r'Classification.json'

# Call the make_json function
make_json(csvFilePath, jsonFilePath)
