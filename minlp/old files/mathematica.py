import subprocess

mathKernel = r"C:\Program Files\Wolfram Research\Mathematica\11.0\MathKernel.exe"
mathFile = r"C:\Users\pu\Desktop\test.m"
run_script = r'"{}" -initfile "{}"'.format(mathKernel, mathFile)

process=subprocess.Popen(run_script,
        shell=True,
        stdout=subprocess.PIPE)

import code;code.interact(local=locals())