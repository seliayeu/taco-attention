for sparsity in $(seq 0.5 0.05 0.95)
do
    # LD_LIBRARY_PATH=../taco/build/lib/ python tests/run.py ./a.out \
    #     --n_q 512 --n_k 512 --d_k 1024 --d_v 1024 --sparsity $sparsity --QFormat DD --KFormat DD --VFormat DD --PFormat DD --runs 10 \
    #     >> tests/baseline.txt

    LD_LIBRARY_PATH=../taco/build/lib/ python tests/run.py ./a.out \
        --n_q 512 --n_k 512 --d_k 1024 --d_v 1024 --sparsity $sparsity --QFormat CSR --KFormat CSC --VFormat CSC --sample 0 --convert 0 --runs 10 \
        >> tests/csr-csc-csc.txt

    LD_LIBRARY_PATH=../taco/build/lib/ python tests/run.py ./a.out \
        --n_q 512 --n_k 512 --d_k 1024 --d_v 1024 --sparsity $sparsity --QFormat CSR --KFormat CSC --VFormat DD --sample 0 --convert 0 --runs 10 \
        >> tests/csr-csc-dd.txt

    LD_LIBRARY_PATH=../taco/build/lib/ python tests/run.py ./a.out \
        --n_q 512 --n_k 512 --d_k 1024 --d_v 1024 --sparsity $sparsity --QFormat DD --KFormat CSC --VFormat CSC --sample 0 --convert 0 --runs 10 \
        >> tests/dd-csr-csc.txt

    LD_LIBRARY_PATH=../taco/build/lib/ python tests/run.py ./a.out \
        --n_q 512 --n_k 512 --d_k 1024 --d_v 1024 --sparsity $sparsity --QFormat CSR --KFormat DD --VFormat CSC --sample 0 --convert 0 --runs 10 \
        >> tests/csr-dd-csc.txt
done
