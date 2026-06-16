TRAIN_TEST_SPLIT=navmini
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navmini
LOG_FILE=$NAVSIM_EXP_ROOT/metric_caching_log.txt

echo "=== 开始运行 metric caching ===" > $LOG_FILE
echo "时间: $(date)" >> $LOG_FILE
echo "TRAIN_TEST_SPLIT: $TRAIN_TEST_SPLIT" >> $LOG_FILE
echo "CACHE_PATH: $CACHE_PATH" >> $LOG_FILE

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    metric_cache_path=$CACHE_PATH \
    force_feature_computation=True 2>&1 | tee -a $LOG_FILE

echo "=== 运行结束 ===" >> $LOG_FILE
echo "时间: $(date)" >> $LOG_FILE