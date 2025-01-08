import os
import subprocess

def test_llm(inputExe, inputFolder, inputJson):
    # Step 1: 递归查找指定文件夹inputFolder下的指定文件inputJson
    for root, dirs, files in os.walk(inputFolder):
        if inputJson in files:
            json_path = os.path.join(root, inputJson)
            break
    else:
        raise FileNotFoundError(f"File {inputJson} not found in {inputFolder}")

    # Step 2: 运行可执行程序inputExe, 将查找到的文件inputJson作为参数传入
    command = [inputExe, '--inputJson', json_path]
    result = subprocess.run(command, capture_output=True, text=True)

    # Step 3: 将运行可执行程序inputExe的输出结果作为返回值
    return result.stdout


if __name__ == "__main__":
    path_case = "D:\\test_phaseCases\\PPM_test\\XHP094-3321714"
    path_bin = "D:\\Atreat\\AtreatLites\\50\\PhaseProgressMonitoring\\bin\\PhaseProgressMonitoring.exe"
    test_llm(path_bin, path_case, "taskInfo.json")
