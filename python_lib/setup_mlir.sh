# alias sed=/opt/homebrew/opt/gsed/bin/gsed
pushd qss_compiler/mlir/dialects
# edit _gen_files
for a in `ls *_ops_gen.py`; do 
    sed -i 's/._ods_common/mlir.dialects._ods_common/g' $a; 
done

# link so libraries
# mkdir _mlir_libs
# cd _mlir_libs
# ln -s ../../../../../../lib/_ibmDialectsPulse.cpython-39-darwin.so 
# ln -s ../../../../../../lib/_ibmDialectsQUIR.cpython-39-darwin.so 
# ln -s ../../../../../../lib/_ibmDialectsQCS.cpython-39-darwin.so 
# ln -s ../../../../../../lib/_ibmDialectsOS3.cpython-39-darwin.so 
# popd