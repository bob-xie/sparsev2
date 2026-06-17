export HYDRA_FULL_ERROR=1

config=default_training
agent=sparsedrive_agent

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name $config \
    agent=$agent \
    experiment_name=$agent \
    train_test_split=navmini \
    use_cache_without_dataset=True  \
    force_cache_computation=False \
    cache_path=exp/data_cache_navmini \
    dataloader.params.batch_size=8 \
    dataloader.params.num_workers=4 \
    dataloader.params.prefetch_factor=2 \
    trainer.params.max_epochs=10 \
    trainer.params.accelerator=gpu \
    trainer.params.strategy=ddp \
    agent.lr=0.0001
