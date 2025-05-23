for sparsity in $(seq 0.5 0.05 0.95)
do
    LD_LIBRARY_PATH=../taco/build/lib/ python scripts/run.py ./a.out \
        --n_q 512 --n_k 512 --d_k 1024 --d_v 1024 --sparsity $sparsity --QFormat DD --KFormat DD --VFormat DD --sample 1 --convert 0 --runs 10 \
        >> scripts/results/baseline-1-0.txt

    LD_LIBRARY_PATH=../taco/build/lib/ python scripts/run.py ./a.out \
        --n_q 512 --n_k 512 --d_k 1024 --d_v 1024 --sparsity $sparsity --QFormat CSR --KFormat CSC --VFormat CSC --sample 1 --convert 1 --runs 10 \
        >> scripts/results/csr-csc-csc-1-1.txt

    LD_LIBRARY_PATH=../taco/build/lib/ python scripts/run.py ./a.out \
        --n_q 512 --n_k 512 --d_k 1024 --d_v 1024 --sparsity $sparsity --QFormat CSR --KFormat CSC --VFormat DD --sample 1 --convert 1 --runs 10 \
        >> scripts/results/csr-csc-dd-1-1.txt

    LD_LIBRARY_PATH=../taco/build/lib/ python scripts/run.py ./a.out \
        --n_q 512 --n_k 512 --d_k 1024 --d_v 1024 --sparsity $sparsity --QFormat DD --KFormat CSC --VFormat CSC --sample 1 --convert 1 --runs 10 \
        >> scripts/results/dd-csr-csc-1-1.txt

    LD_LIBRARY_PATH=../taco/build/lib/ python scripts/run.py ./a.out \
        --n_q 512 --n_k 512 --d_k 1024 --d_v 1024 --sparsity $sparsity --QFormat CSR --KFormat DD --VFormat CSC --sample 1 --convert 1 --runs 10 \
        >> scripts/results/csr-dd-csc-1-1.txt
done
