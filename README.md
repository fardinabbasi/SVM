# Support Vector Machine
## SVC
Performing **Support Vector Classification (SVC)** on the [iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html), which comprises **three classes** of flower types: 'setosa', 'versicolor', and 'virginica', each containing 50 samples. The dataset consists of **four features**: Sepal Length, Sepal Width, Petal Length, and Petal Width.
### Preprocessing
The dataset has been divided into a **training set** and a **test set**.
### Linear Kernel
The linear kernel should be employed when the datapoints are **linearly separable**, given its lower number of **parameters** to learn, resulting in **faster computation**, making it particularly well-suited for **high-dimensional datasets**.

Below are the classification results using the **linear kernel**, first with the features Sepal Length & Sepal Width, and then with the features Petal Length & Petal Width.
| Result | Sepal Length & Sepal Width | Petal Length & Petal Width |
| --- | --- | --- |
| Decision Regions | <img src="/readme_images/s_p.png"> | <img src="/readme_images/p_p.png"> |
| Confusion Matrix | <img src="/readme_images/s_c.png"> | <img src="/readme_images/p_c.png"> |
| Classification Report | <img src="/readme_images/s_r.jpg"> | <img src="/readme_images/p_r.jpg"> |

Based on the results obtained using the petal length & petal width features, where datapoints exhibit **linear separability**, the classification performance is notably impressive.
### Poly and RBF Kernel
1. **RBF Kernel**: RBF kernel is the most **generalized** form of kernelization and is one of the most **widely used** kernels due to its similarity to the **Gaussian distribution**. [Read More](https://towardsdatascience.com/radial-basis-function-rbf-kernel-the-go-to-kernel-acf0d22c798a)

	The RBF kernel function for two points $X_1$ and $X_2$ computes the **similarity** or how close they are to each other. This kernel can be mathematically represented as follows:

$$K(X_1, X_2) = \exp\left(-\frac{\|X_1 - X_2\|^2}{2\sigma^2}\right)$$
2. **Polynomial Kernel**: In general, the polynomial kernel is defined as:

$$K(X_1, X_2) = (X_1^TX_2+a)^b$$
Below are the classification results using the **poly and RBF kernel** with all 4 features.
| Result | Kernel = Poly | Kernel = RBF |
| --- | --- | --- |
| Confusion Matrix | <img src="/readme_images/poly_c.png"> | <img src="/readme_images/rbf_c.png"> |
| Classification Report | <img src="/readme_images/rbf_r.jpg"> | <img src="/readme_images/rbf_r.jpg"> |

## Hyperparameter Tuning
The optimization function for **Soft SVM** is written as follows:

$$
\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i
$$

subject to:

$$
\begin{align*}
& y_i(w^T x_i + b) \geq 1 - \xi_i, \quad i = 1, 2, \ldots, n \\
& \xi_i \geq 0, \quad i = 1, 2, \ldots, n
\end{align*}
$$

**C** is a **hyperparameter** which determines the **trade-off** between lower error or higher **margin**.

To determine the **optimal** 'C' value for each kernel and the best 'gamma' value for the RBF kernel, **GridSearchCV** is utilized as shown below.
```ruby
from sklearn.model_selection import GridSearchCV
```
```ruby
param = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
kernels = ['linear', 'poly', 'rbf']
best_param={}

for kernel in kernels:
	model = SVC(kernel=kernel)
	grid_search = GridSearchCV(model, param_grid=param)
	grid_search.fit(X_train, y_train)
	best_param[kernel] = grid_search.best_params_
	print("Optimal hyperparameters for Kernel = "+str(kernel)+ ":"+str(best_param[kernel]))
	print("Accuracy on test set for Kernel = "+str(kernel)+ ":" +str(grid_search.score(X_test, y_test)*100)+"%\n")
```

*Optimal hyperparameters for Kernel = linear:{'C': 1, 'gamma': 0.1}*

*Optimal hyperparameters for Kernel = poly:{'C': 0.1, 'gamma': 0.1}*

*Optimal hyperparameters for Kernel = rbf:{'C': 100, 'gamma': 0.01}*

Since the dataset is multi-class, two methods, namely **One Vs. Rest** and **One Vs. One**, are employed to classify the data. 
The "[decision_function_shape](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)" parameter is set accordingly. Here are the classification results obtained for each method.
### One Vs. Rest
| Result | Kernel = Linear | Kernel = Poly | Kernel = RBF |
| --- | --- | --- | --- |
| Confusion Matrix | <img src="/readme_images/or_linear_c.png"> | <img src="/readme_images/or_poly_c.png"> | <img src="/readme_images/or_rbf_c.png"> |
| Classification Report | <img src="/readme_images/or_linear_r.jpg"> | <img src="/readme_images/or_poly_r.jpg"> | <img src="/readme_images/or_rbf_r.jpg"> |

### One Vs. One
| Result | Kernel = Linear | Kernel = Poly | Kernel = RBF |
| --- | --- | --- | --- |
| Confusion Matrix | <img src="/readme_images/oo_linear_c.png"> | <img src="/readme_images/oo_poly_c.png"> | <img src="/readme_images/oo_rbf_c.png"> |
| Classification Report | <img src="/readme_images/oo_linear_r.jpg"> | <img src="/readme_images/oo_poly_r.jpg"> | <img src="/readme_images/oo_rbf_r.jpg"> |

In this particular problem, the One Vs. One method exhibits slightly better classification performance.
