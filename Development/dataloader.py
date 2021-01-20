import os
import shutil
        
def copy_file(src, dest):
    "Copy file from source dir to dest dir. Note that the path must exist where folders should be placed!"
    assert os.path.exists(src), "Source file does not exists"
    
    if not os.path.exists(dest):
        shutil.copy(src, dest)
        return 'ok' 
    return "fail"