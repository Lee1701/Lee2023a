All our deep learning models are based on the DeepPurpose library,
https://github.com/kexinhuang12345/DeepPurpose.

Our codes were modified and customized from the DeepPurpose library v0.0.1.

## File description
1. `dp.pretrained.models.v015.bdb`<br>
  : Prediction of PDBbind by the pretrained models with BindingDB from DeepPurpose v0.1.5
2. `training.bindingdb.full.data`<br>
  : Training on BindingDB full data <i>de novo</i>
3. `finetuning.bdb.fulldata.pdbbind2020.refined.excore`<br>
  : Training on BindingDB full data and then finetuning on PDBbind-RefinedSet (excluding the CoreSet)
4. `training.pdbbind2020.refined.regsm.excore.5fcv.10reps`<br>
  : Training with PDBbind2020-RefinedSet (excluding the CoreSet) by 10 repeted 5-fold cross-validations
