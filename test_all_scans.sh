# declare -a arr=(27355 27299 27303 27285 27307 27296 27305 27289 27282 27287 27294 27290 27325 27319 27347 27340 27338 27336 27326)


declare -a arr=(27282)
# Multidirect 
for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp_simple --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 1200 --step_size 10 --batch_size 200 --multidirect
done

