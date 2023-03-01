# hearthstone-vpt

In progress, doing [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://arxiv.org/abs/2206.11795) in Hearthstone

## Currently:

- Scripts for collecting supervised training data from windows hearthstone client in *collect_data*
- using trained IDMs for lableing unlabled YouTube data, training policies, running trained policies in hearthstone cilent, 

## TODO:

- Collect data, possibly try finetuning from pre-trained Minecraft policies, 
- Redo state/action parameterization to match VPT, (ie discretize mouse movement, etc)
- Better documentation 
