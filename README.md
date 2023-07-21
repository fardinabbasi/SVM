# SVM

| Result | Sepal Length & Sepal Width | Petal Length & Petal Width |
| --- | --- | --- |
| Decision Regions | <img src="/readme_images/s_p.png"> | <img src="/readme_images/p_p.png"> |
| Confusion Matrix | <img src="/readme_images/s_c.png"> | <img src="/readme_images/p_c.png"> |
| Classification Report | <img src="/readme_images/s_r.jpg"> | <img src="/readme_images/p_r.jpg"> |

| Result | Kernel = Poly | Kernel = RBF |
| --- | --- | --- |
| Confusion Matrix | <img src="/readme_images/poly_c.png"> | <img src="/readme_images/rbf_c.png"> |
| Classification Report | <img src="/readme_images/rbf_r.jpg"> | <img src="/readme_images/rbf_r.jpg"> |

## Hyperparameter Tuning
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
