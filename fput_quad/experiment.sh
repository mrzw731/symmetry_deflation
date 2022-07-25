for N_sites in {3..10} #3 4 5 6 7 8 9 10
do
    python fput_quad_new.py --num_Hs $((2*N_sites)) --N_sites $N_sites --epochs 2 --deflate 1
done
