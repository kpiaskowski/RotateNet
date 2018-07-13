import os
import shutil

summaries_path = '../summaries'
models_path = '../saved_models'

shutil.rmtree(summaries_path, ignore_errors=True)
shutil.rmtree(models_path, ignore_errors=True)

os.mkdir(summaries_path)
os.mkdir(models_path)
