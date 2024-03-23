import csv

input_file = 'articles_data2.csv'
output_file = 'urls.csv'

# Function to extract URLs from a CSV file
def extract_urls(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_in, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_out:
        reader = csv.DictReader(csv_in)
        writer = csv.writer(csv_out)
        
        # Write header to output file
        writer.writerow(['URL'])
        
        # Iterate over rows and extract URLs
        for row in reader:
            url = row.get('URL')
            if url:
                writer.writerow([url])

# Call the function to extract URLs
extract_urls(input_file, output_file)
