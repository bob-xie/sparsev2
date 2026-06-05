python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=sparsedrive_agent \
    experiment_name=cache_navmini \
    train_test_split=navmini \
    cache_path=$NAVSIM_EXP_ROOT/data_cache_navmini \
    worker=sequential   # ← 手动添加