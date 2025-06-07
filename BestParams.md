'preprocessor', EnhancedImagePreprocessor(
            use_edges=True,
            use_histogram_eq=True,
            use_clahe=True,
            use_color_spaces=True,
            use_lbp=True,
            use_gabor=False,
            use_advanced_edges=True,
            use_morphology=False,
            use_hog=True,
            use_superpixels=False,
            normalize_features=True
        )),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=150, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=3, weights='distance', p=2)






Best parameters: {'knn__n_neighbors': 4, 'knn__p': 2, 'knn__weights': 'distance', 'pca__n_components': 50, 'preprocessor__use_advanced_edges': False, 'preprocessor__use_clahe': True, 'preprocessor__use_edges': True, 'preprocessor__use_gabor': False, 'preprocessor__use_histogram_eq': True, 'preprocessor__use_hog': False, 'preprocessor__use_lbp': False, 'preprocessor__use_morphology': False}
    