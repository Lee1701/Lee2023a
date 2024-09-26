All our deep learning models are based on the DeepPurpose library,
https://github.com/kexinhuang12345/DeepPurpose.

Our codes were modified and customized from the DeepPurpose library v0.0.1. We provide example codes for selected models in this repository.

## File description
1. `dp.pretrained.models.v015.bdb`<br>
  : Prediction of PDBbind by the pretrained models with BindingDB from DeepPurpose v0.1.5 (D1 models)
2. `training.bindingdb.full.data`<br>
  : Training on BindingDB full data <i>de novo</i> (D2 models)
3. `finetuning.bdb.fulldata.pdbbind2020.refined.excore`<br>
  : Training on BindingDB full data and then finetuning on PDBbind v2020 RefinedSet\CoreSet (D2F models)
4. `training.pdbbind2020.refined.regsm.excore.5fcv.10reps`<br>
  : Training with PDBbind v2020 RefinedSet\CoreSet by 10 repeted 5-fold cross-validations (D3 models)
