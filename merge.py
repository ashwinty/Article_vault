# import json

# # Function to rearrange fields in each entry
# def rearrange_fields(entry):
#     return {
#         "Title": entry["Title"],
#         "Tag": entry["Tag"],
#         "Author": entry["Author"],
#         "Date": entry["Date"],
#         "Article URL": entry["Article URL"],
#         "Description": entry["Description"],
#         "Main Image URL": entry["Main Image URL"],
#         "Article Text": entry["Article Text"]
#     }

# # Read JSON file
# with open("modified_json_file.json", "r", encoding="utf-8") as file:
#     data = json.load(file)

# # Rearrange fields for each entry
# rearranged_data = [rearrange_fields(entry) for entry in data]

# # Write rearranged data back to a file
# with open("rearranged_json_file.json", "w", encoding="utf-8") as file:
#     json.dump(rearranged_data, file, indent=2, ensure_ascii=False)

# print("Rearranged JSON data written to rearranged_json_file.json")











# import json

# # Load the JSON data from file
# with open('merged_data.json', 'r') as file:
#     data = json.load(file)

# # Iterate through each entry in the JSON data
# for entry in data:
#     # Change the tag names
#     entry['Article URL'] = entry.pop('URL')
#     entry['Description'] = entry.pop('Text')

# # Write the modified data back to the file
# with open('modified_json_file.json', 'w') as file:
#     json.dump(data, file, indent=2)













# import json
# import csv

# # Load JSON data
# with open('extracted_data.json', 'r') as json_file:
#     json_data = json.load(json_file)

# # Load CSV data
# csv_data = []
# with open('articles_data.csv', 'r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     for row in csv_reader:
#         csv_data.append(row)

# # Merge JSON and CSV data
# merged_data = []
# for json_item, csv_item in zip(json_data, csv_data):
#     merged_item = {
#         "Title": csv_item["Title"],
#         "URL": csv_item["URL"],
#         "Text": csv_item["Text"],
#         "Tag": csv_item["Tag"],
#         "Author": csv_item["Author"],
#         "Date": csv_item["Date"],
#         "Main Image URL": json_item["Main Image URL"],
#         "Article Text": json_item["Article Text"]
#     }
#     merged_data.append(merged_item)

# # Write merged data to JSON file
# with open('merged_data.json', 'w') as outfile:
#     json.dump(merged_data, outfile, indent=2)
