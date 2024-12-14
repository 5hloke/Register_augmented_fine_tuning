# Register_augmented_fine_tuning

This project was undertaken as part of the course EECS 598: Large Language Models, at the University of Michigan, Ann Arbor, under the guidance of Professor Samet Oymak. We look at [Register-Augmented Fine-Tuning](https://arxiv.org/abs/2309.16588), a method that was introduced for Vision Transformers, and try to apply it to a language context, on the Question Answering task. We use [Layer-wise Relevance Propagation](https://arxiv.org/abs/2012.09838) and [Integrated Gradients](https://arxiv.org/abs/1703.01365) to analyze the attention map for a qualitative study, and use F1 score and ExactMatch for a quantitative study. We use the [TyDi QA Dataset](https://drive.google.com/drive/folders/1cBCIViRZ38zBlnn2c0UUMJHoQddClw6y?usp=share_link) for our work. 

To reproduce our work, run the following to instantiate a Conda environment with all our required dependencies.

```
conda env create -f environment.yml --name myenv
conda activate myenv
pip install -r requirements.txt
```

Once the environment has been instantiated, you can run `QA_script.py` to perform an ablation study on the number of registers. `integrated_grad_vis.ipynb` and `lrp_vis.ipynb` can be run to visualize the LRP and Integrated Gradients attention analysis.