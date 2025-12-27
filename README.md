# SVM From Scratch (Dual Formulation)

This project implements a soft-margin **linear Support Vector Machine (SVM)** from scratch
using the dual formulation and `scipy.optimize`.  
I then verified it visually on a toy dataset and evaluated it on the Breast Cancer dataset,
comparing the results against scikit-learn’s SVM.

---

## What’s Inside

- `notebooks/01_toy_linear_svm.ipynb`  
  Derives and visualizes the SVM decision boundary, margins, and support vectors.

- `notebooks/02_real_dataset_experiments.ipynb`  
  Trains the custom SVM on the Breast Cancer dataset, compares it to scikit-learn,
  and analyzes how the regularization parameter **C** affects accuracy and support vectors.

---

## 1. Toy Dataset — Geometry & Intuition

Using a simple 2-D “customer engagement” dataset, the solver finds:

- only **2 of 4** points become support vectors  
- margin constraints are satisfied exactly  
- the boundary sits midway between support vectors

![toy plot](images/toy_decision_boundary.png)

The solid line is the decision boundary.  
The dashed lines are the margins.  
Circled points are the support vectors — the only samples that actually define the boundary.

---

## 2. Real Dataset — Model Behavior

On the Breast Cancer dataset, I trained:

- my custom SVM
- scikit-learn’s `SVC(kernel="linear")`

Both models achieved identical **training accuracy (0.991)** and used the same number of support vectors (**32**).

Test accuracy was comparable:
Custom SVM: 0.947
scikit-learn: 0.974

## Effect of Regularization (C)

I swept several values of C and observed how the model changes.

### Test Accuracy vs C
![accuracy plot](images/c_sweep_accuracy_support_vectors.png)

Accuracy is high across a wide range of C values,
showing the dataset is fairly separable after standardization.

### Support Vectors vs C
![support vectors plot](images/c_sweep_sv_only.png)

As expected:

- **small C** → wide margin, **more** support vectors  
- **large C** → narrow margin, **fewer** support vectors  

This matches SVM theory.

---

## Key Takeaways

- Implemented SVM from scratch using the **dual optimization problem**
- Reconstructed the primal parameters \(w\) and \(b\)
- Visualized margins and support vectors
- Matched scikit-learn behavior on a real dataset
- Explored how **C** controls the trade-off between margin size and misclassification

---

## Running

Install dependencies:

```bash
pip install numpy scipy matplotlib scikit-learn
```

Then open the notebooks:
```bash
jupyter lab notebooks/
```

--- 

## License
This project is licensed under the [MIT License](LICENSE)
