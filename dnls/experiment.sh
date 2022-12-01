for N_sites in {3..10} #3 4 5 6 7 8 9 10
do
    python dnls_involution.py --num_Hs $((2*N_sites)) --N_sites $N_sites --train_box 1000 --epochs 5000 --deflate .5
done
