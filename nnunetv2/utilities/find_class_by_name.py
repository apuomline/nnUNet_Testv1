import importlib
import pkgutil

from batchgenerators.utilities.file_and_folder_operations import *

"""
在指定路径下，递归寻找目标模型---有许多类需要获取--编写一个通用函数可以获取想要的类
可能是由于 需要根据配置信息等等--程序自动选取需要的类
(哪些地方需要自动选取需要的类)

"""

def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr