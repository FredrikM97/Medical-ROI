"""
Created at 07.11.19 19:12
@author: gregor

"""

import os, sys, site
from pathlib import Path

# recognise newly installed packages in path
site.main()

from setuptools import setup,find_packages
from torch.utils import cpp_extension

dir_ = Path(os.path.dirname(sys.argv[0]))

setup(name='RoIAlign_extension',
      ext_modules=[
          cpp_extension.CUDAExtension('RoIAlign_extension_3d', [str(dir_/'3D/RoIAlign_interface_3d.cpp'),
                                                                       str(dir_/'3D/RoIAlign_cuda_3d.cu')]
                                              ), 
           cpp_extension.CUDAExtension('RoIAlign_extension_2d', [str(dir_/'2D/RoIAlign_interface.cpp'),
                                                        str(dir_/'2D/RoIAlign_cuda.cu')]
                                      ),
          ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
    )
