from .dhvr_training import DHVRTraining

modules = [DHVRTraining]
modules_dict = {m.__name__: m for m in modules}


def get_training_module(module_name):
    assert (
        module_name in modules_dict.keys()
    ), f"{module_name} not in {modules_dict.keys()}"
    return modules_dict[module_name]
