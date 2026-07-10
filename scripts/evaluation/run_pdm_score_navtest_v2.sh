export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT=navmini
CHECKPOINT=exp/sparsedrive_agent/2026.06.17.17.46.52/periodic_pdm_ckpts/ep0010.ckpt
CACHE_PATH=exp/metric_cache_navmini

python3 $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_navtest_v2_fast.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=sparsedrive_agent \
    agent.checkpoint_path=$CHECKPOINT \
    experiment_name=sparsedrive_agent \
    metric_cache_path=$CACHE_PATH \
    +test_cache_path=exp/data_cache_navmini \
    dataloader.params.batch_size=8
