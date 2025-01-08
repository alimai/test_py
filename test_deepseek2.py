
import os
import json

def generate_taskInfo(inputFolder, outputFileName):
    def find_files(directory, extension):
        return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(extension)]

    def check_conditions(directory):
        ods_files = find_files(directory, ".ods")
        ddm_files = find_files(directory, ".ddm")
        return len(ods_files) >= 1 and len(ddm_files) > 1

    def process_directory(directory):
        if check_conditions(directory):
            output_path = os.path.join(directory, outputFileName)
            with open(output_path, 'w') as f:
                json.dump({"status": "success"}, f)
            return output_path
        return None

    for root, dirs, _ in os.walk(inputFolder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            result = process_directory(dir_path)

            if result:
                print(result)

def generate_taskInfo2(inputFolder, outputFileName):
    def find_files(directory, extension):
        return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(extension)]

    def check_conditions(directory):
        ods_files = find_files(directory, ".ods")
        ddm_files = find_files(directory, ".ddm")
        return len(ods_files) >= 1 and len(ddm_files) > 1

    def process_directory(directory):
        if check_conditions(directory):
            ods_files = find_files(directory, ".ods")
            ddm_files = find_files(directory, ".ddm")
            output_path = os.path.join(directory, outputFileName)
            with open(output_path, 'w') as f:
                json.dump({"ods_files": ods_files, "ddm_files": ddm_files}, f)
            return output_path
        return None

    for root, dirs, _ in os.walk(inputFolder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            result = process_directory(dir_path)

            if result:
                print(result)

if __name__ == "__main__":
    result = generate_taskInfo2("D:\\test_phaseCases\PPM_test\BUG_Arch_Value_0", "taskInfo2.json")
