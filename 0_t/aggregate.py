import os
import shutil

tsv_files_list = os.listdir('res')

for file in tsv_files_list:
    file_split = file.split('.')

    if len(file_split) == 5:
        app = file_split[1]
    else:
        app = file_split[0]

    model = file_split[-2]

    directory_path = os.path.join('res', app, model)
    os.makedirs(directory_path, exist_ok=True)

    file_path = os.path.join('res', file)
    new_file_path = os.path.join(directory_path, file)
    shutil.move(file_path, new_file_path)