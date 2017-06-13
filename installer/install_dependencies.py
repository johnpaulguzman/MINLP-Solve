import os
import pip

current_dir = os.path.split(os.path.abspath(__file__))[0]
dependencies_dir = current_dir + "\\..\\dependencies"
requirements_path = current_dir + "\\requirements.txt"
wheel_files = os.listdir(dependencies_dir)
wheel_files.sort()
for whl in wheel_files:
    pip.main(["install", dependencies_dir + "\\" + whl])
pip.main(["install", "-r", requirements_path])