declare -a arr=(27355 27299 27303 27285 27307 27296 27305 27289 27282 27287 27294 27290 27325 27319 27347 27340 27338 27336 27326)

# Multidirect 
for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp1 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 500 --step_size 10 --batch_size 200 --multidirect --use_loftr --loftr_conf_thresh 0.85
done

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp2 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 1000 --step_size 10 --batch_size 200 --multidirect --use_loftr --loftr_conf_thresh 0.85
done

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp3 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 1300 --step_size 10 --batch_size 200 --multidirect --use_loftr --loftr_conf_thresh 0.85
done


# Unidirect

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp4 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 300 --step_size 10 --batch_size 200 --use_loftr --loftr_conf_thresh 0.85
done

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp5 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 400 --step_size 10 --batch_size 200 --use_loftr --loftr_conf_thresh 0.85
done

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp6 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 150 --step_size 10 --batch_size 200 --use_loftr --loftr_conf_thresh 0.85
done


# No LOFTR

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp7 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 300 --step_size 10 --batch_size 200
done

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp8 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 400 --step_size 10 --batch_size 200 
done

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp9 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 150 --step_size 10 --batch_size 200
done

# LOFTR

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp10 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 300 --step_size 10 --batch_size 200 --use_loftr
done

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp11 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 400 --step_size 10 --batch_size 200 --use_loftr
done

for i in "${arr[@]}"
do
    echo "Test scan sample $i"
    python test_stitcher.py --exp_name exp12 --model ~/trained_weights/metric/traced/best_emb.pt --source dataset/stitching_$i/ --emb_input_size 400 --window_size 150 --step_size 10 --batch_size 200 --use_loftr
done