# Creating files train.txt and test.txt
# for training in Darknet framework

# Importing needed library
import os

# Full or absolute path to the folder with Traffic Signs images

full_path_to_images = 'C://Users//DELL//darknet//data//images'



# Getting list of full paths to downloaded images

# Changing the current directory
os.chdir(full_path_to_images)

# Defining list to write paths in
p = []

for current_dir, dirs, files in os.walk('.'):
    # Going through all files
    for f in files:
        # Checking if filename ends with '.jpg'
        if f.endswith('.jpg'):
            # Preparing path to save into train.txt file

            path_to_save_into_txt_files = full_path_to_images + '/' + f

            p.append(path_to_save_into_txt_files + '\n')


# Slicing first 15% of elements from the list
# to write into the test.txt file
p_val = p[:int(len(p) * 0.15)]

# Deleting from initial list first 15% of elements
p = p[int(len(p) * 0.15):]


# Creating file train.txt and writing 85% of lines in it

with open('train.txt', 'w') as train_txt:
    # Going through all elements of the list
    for e in p:
        # Writing current path at the end of the file
        train_txt.write(e)

# Creating file test.txt and writing 15% of lines in it
with open('val.txt', 'w') as val_txt:
    # Going through all elements of the list
    for e in p_val:
        # Writing current path at the end of the file
        val_txt.write(e)
