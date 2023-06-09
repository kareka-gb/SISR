## Note about the saved models

All models are saved in the following python `.pkl` format

Optimizer used - Adam
Scheduler used - StepLR
Loss used - L1Loss

```

{
'epoch': 100, # epoch end
'model_state_dict': model.state_dict(),
'optimizer_state_dict': optimizer.state_dict(),
'lr': 0.0001,
'scheduler_state_dict': scheduler.state_dict(),
'gamma': 0.8577,
'loss': 'L1Loss',
}

```