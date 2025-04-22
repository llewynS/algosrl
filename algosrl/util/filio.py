import os

def get_exp_path(path_to_folder, my_class_name):
    # Ensure the main folder exists
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)

    # List all directories in the main folder
    dirs = [d for d in os.listdir(path_to_folder) if os.path.isdir(os.path.join(path_to_folder, d))]

    # Filter directories that match the class name pattern and get their numbers
    class_nums = [int(d.split('_')[-1]) for d in dirs if d.startswith(my_class_name + '_') and d.split('_')[-1].isdigit()]

    # Find the next class number
    next_class_num = max(class_nums, default=0) + 1

    # Create the new directory
    new_dir_path = os.path.join(path_to_folder, f"{my_class_name}_{next_class_num}")
    os.makedirs(new_dir_path)

    return new_dir_path