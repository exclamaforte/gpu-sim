import sys, importlib.util
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("pv_old", "/tmp/pv_old.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
v = m.PreflightValidator(arch="sm_120")
# the pre-fix stub, called directly on a config it never ran:
print("PRE-FIX  _run_under_fakegpu ->", v._run_under_fakegpu(
    {"block_x": 999999}, "a_kernel_that_does_not_exist"))
print("PRE-FIX  same call, empty config ->", v._run_under_fakegpu({}, "nope"))
