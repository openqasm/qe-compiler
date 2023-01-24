import os

LLVM_TAG = "14.0.6"


def get_or_update_llvm(conanfile):
    if not os.path.exists("llvm-project/.git"):
        # Sources not yet downloaded.
        conanfile.run(f"git clone -b git@github.com:llvm/llvm-project.git")

    # Check out LLVM at correct tag. This will fail if you have local changes
    # to llvm-project.
    conanfile.run(f"git checkout {LLVM_TAG}")
