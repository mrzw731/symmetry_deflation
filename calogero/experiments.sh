# large box = 100000

for N_sites in {5..10} #5 6 7 8 9 10
do
    python calogero_involution.py --num_Hs $((2*N_sites)) --N_sites $N_sites --train_box 10000 --epochs 15000 --deflate 1 --width 800
done

