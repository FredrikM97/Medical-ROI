"""
Created at 07.11.19 19:12
@author: gregor

"""

import os, sys, site
from pathlib import Path

# recognise newly installed packages in path
site.main()

from setuptools import setup, find_packages
from torch.utils import cpp_extension

dir_ = Path(os.path.dirname(sys.argv[0]))


setup(name='nms_extension',
      ext_modules=[
          cpp_extension.CUDAExtension('nms_extension', 
                                       sources=[str(dir_/'core/nms_interface.cpp'), str(dir_/'core/nms.cu')],
                                      ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      )
