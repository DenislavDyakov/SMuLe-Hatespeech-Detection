from csv import writer
from csv import reader

import googletrans
from googletrans import Translator

translator = Translator()
print("Start translation")
default_text = 'Some Text'
# Open the input_file in read mode and output_file in write mode
with open('jigsaw-toxic-comment-train.csv', 'r', encoding="utf8", errors='ignore') as read_obj, \
        open('jigsaw-toxic-comment-train-BG.csv', 'w', newline='', encoding="utf8", errors='ignore') as write_obj:
    # Create a csv.reader object from the input file object
    csv_reader = reader(read_obj)
    # Create a csv.writer object from the output file object
    csv_writer = writer(write_obj)
    # Read each row of the input csv file as list
    for row in csv_reader:
        # print(row[1])
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # Append the default text in the row / list
        row.append(translator.translate(row[1], dest='bg').text)
        # Add the updated row / list to the output file
        csv_writer.writerow(row)
print("End translation")