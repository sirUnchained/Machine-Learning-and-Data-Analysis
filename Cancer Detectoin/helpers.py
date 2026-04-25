from pathlib import Path
import os
import warnings
import importlib


def optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        warnings.warn(f"{module_name} is not installed. Some helpers will not work.")
        return None


tf = optional_import("tensorflow")
np = optional_import("numpy")
pd = optional_import("pandas")

plt = optional_import("matplotlib.pyplot")
mdates = optional_import("matplotlib.dates")

sk_metrics = optional_import("sklearn.metrics")
sk_model_selection = optional_import("sklearn.model_selection")

cv2 = optional_import("cv2")
a = optional_import("albumentations")


if sk_metrics:
    roc_curve = sk_metrics.roc_curve
    classification_report = sk_metrics.classification_report
    accuracy_score = sk_metrics.accuracy_score
    precision_score = sk_metrics.precision_score
    recall_score = sk_metrics.recall_score
    f1_score = sk_metrics.f1_score
    mean_absolute_error = sk_metrics.mean_absolute_error
    mean_squared_error = sk_metrics.mean_squared_error

if sk_model_selection:
    cross_val_score = sk_model_selection.cross_val_score
    KFold = sk_model_selection.KFold
    GridSearchCV = sk_model_selection.GridSearchCV
    StratifiedKFold = sk_model_selection.StratifiedKFold


class ClassificationModelResult:
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    model_name = ""

    def __init__(
        self, accuracy, precision, recall, f1_score: float, model_name: str
    ) -> None:
        self.f1_score = f1_score
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.model_name = model_name


class RegressionModelResult:
    mean_absolute_error = 0.0
    mean_square_error = 0.0
    root_mean_square_error = 0.0
    r2_score = 0.0
    model_name = ""

    def __init__(
        self,
        mean_absolute_error,
        mean_square_error,
        root_mean_square_error,
        r2_score: float,
        model_name: str,
    ) -> None:
        self.mean_absolute_error = mean_absolute_error
        self.mean_square_error = mean_square_error
        self.root_mean_square_error = root_mean_square_error
        self.r2_score = r2_score
        self.model_name = model_name


def load_and_prep_image(filename: Path, img_shape=224, scale=True):
    """
    This function reads an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).

    Param:
        filename (str): string filename of target image
        img_shape (int): size to resize target image to, default 224
        scale (bool): whether to scale pixel values to range(0, 1), default True
    Returns:
        img (tf.tensor): it'll return the image
    """
    # Read in the image
    img = tf.io.read_file(str(filename))
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img / 255.0
    else:
        return img


def plot_images(path: str, count: int = 4, folder_label: bool = False):
    """
    Plot a grid of images from a specified directory.

    Args:
        path (str): Path to the directory containing images
        count (int): Number of images to display (default: 4)
        folder_label (bool): If True, use folder name as title for all images.
                            If False, use individual filenames as titles (default: False)

    Returns:
        None: Displays a matplotlib figure with the images
    """
    src_path = Path(path)
    files_count = count
    files_in_path = os.listdir(src_path)[:files_count]

    if len(files_in_path) < count:
        files_count = len(files_in_path)
        print(
            f"[INFO] count size you requested is larger than items in path, set counts to `{files_count}`"
        )

    if files_count > 4:
        loop = (files_count + 3) // 4  # Ceiling division to handle remainder
        rows = loop
        cols = min(4, files_count)

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5 * rows))

        # Flatten axes for easier indexing
        if rows > 1:
            axes = axes.flatten()
        else:
            axes = axes

        for i in range(files_count):
            img = load_and_prep_image(src_path / files_in_path[i])

            # Set title based on folder_label parameter
            if folder_label:
                # Get parent folder name
                parent_folder = src_path.name
                title_text = parent_folder
            else:
                title_text = files_in_path[i]

            if rows > 1:
                axes[i].imshow(img)
                axes[i].set_title(title_text)
                axes[i].axis("off")
            else:
                axes[i].imshow(img)
                axes[i].set_title(title_text)
                axes[i].axis("off")

        # Hide empty subplots if any
        if rows > 1:
            for i in range(files_count, rows * cols):
                axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    else:
        fig, axes = plt.subplots(ncols=files_count, figsize=(15, 5))

        # Handle case when there's only one subplot
        if files_count == 1:
            axes = [axes]

        for i in range(files_count):
            img = load_and_prep_image(src_path / files_in_path[i])

            # Set title based on folder_label parameter
            if folder_label:
                # Get parent folder name
                parent_folder = src_path.name
                title_text = parent_folder
            else:
                title_text = files_in_path[i]

            axes[i].imshow(img)
            axes[i].set_title(title_text)
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()


def run_grid_search(
    model: object,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    model_name: str,
):
    """
    Performs hyperparameter tuning for a given scikit-learn model using GridSearchCV
    with negative log loss as the scoring metric. This function is designed to work
    for both binary and multiclass classification problems, provided the model
    supports `predict_proba`.

    Parameters
    ----------
    model : object
        An unfitted model instance from scikit-learn (or compatible API)
        that has `fit`, `predict_proba`, `get_params`, and `set_params` methods.
        Example: LogisticRegression(), RandomForestClassifier()

    param_grid : dict or list of dicts
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in order to have the GridSearchCV explore the
        parameter space.
        Example: {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

    X : array-like of shape (n_samples, n_features)
        Training data features.

    y : array-like of shape (n_samples,)
        Training data target labels. This should be a 1D array containing class labels.

    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - integer, to specify the number of folds in a (Stratified)KFold.
        - An object that implements the 'split' method (e.g., KFold, StratifiedKFold).
        Example: 5, or KFold(n_splits=5, shuffle=True, random_state=42)

    model_name : str
        A descriptive name for the model being evaluated, used for printing output.
        Example: "Logistic Regression", "Random Forest", "Multiclass Classifier"

    Returns
    -------
    best_model : object
        The model instance fitted with the best parameters found during the grid search.
        This model is trained on the entire provided training dataset (X, y).

    Notes
    -----
    This function performs a Grid Search to find the optimal hyperparameters for a given model.
    It uses 'neg_log_loss' as the scoring metric because GridSearchCV aims to maximize the score,
    while Log Loss is a metric to be minimized. Using the negative Log Loss effectively turns minimization
    into maximization for GridSearchCV. This metric is suitable for both binary and multiclass problems.
    """
    # Setup `GridSearchCV` object
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring="neg_log_loss", n_jobs=-1, verbose=1
    )

    # Fit the grid search to the training data
    grid_search.fit(X, y)

    print(f"\n--- Grid Search for `{model_name}` ---")
    print(f"Best Hyper Parameters: {grid_search.best_params_}")

    # Retrieve the best cross-validated score (which is the negative log loss)
    best_score_neg_log_loss = grid_search.best_score_
    # Convert the negative log loss score to a positive log loss for better interpretation
    best_log_loss = -best_score_neg_log_loss
    print(f"Best Mean Log Loss (CV): {best_log_loss:.4f}")

    # Get the best model found by GridSearchCV, fitted on the entire training dataset
    best_model = grid_search.best_estimator_

    # Return the best model instance
    return best_model


def plot_roc_curve(model, X_test, y_test):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for a classification model.

    Args:
        model: Trained classifier model with predict_proba method
        X_test: Test features
        y_test: True labels for test data

    Returns:
        None: Displays the ROC curve plot
    """
    # Getting probabilities
    y_probs = model.predict_proba(X_test)
    y_probs_positive = y_probs[:, 1]

    # calculate roc curve
    fpr, tpr, _ = roc_curve(y_test, y_probs_positive)

    # Plot ROC curve
    plt.plot(fpr, tpr, color="orange", label="ROC")
    # Plot the base line
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--", label="Gussing")
    # Plot best AUC score
    t1, t2, _ = roc_curve(y_test, y_test)
    plt.plot(t1, t2, color="green", linestyle="--", label="best score")

    # customize plot
    plt.xlabel("False Positive Rate (fpr)")
    plt.ylabel("True Positive Rate (tpr)")
    plt.title("Reciver Operating Characteristic (ROC) curve")
    plt.legend()
    plt.show()


def evaluate_regression_model(y_pred, y_test, model_name):
    """
    Evaluate a regression model using multiple metrics.

    Args:
        y_pred: model prediction
        y_test: True target values for test data

    Returns:
        dict: Dictionary containing MAE, MSE, RMSE, and R² scores (as percentages)
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return RegressionModelResult(mae, mse, rmse, r2, model_name)
    return {
        "mean absolute error": f"{mae*100:.2f}",
        "mean squared error": f"{mse*100:.2f}",
        "root mean squared error": f"{rmse*100:.2f}",
        "r2 score": f"{r2*100:.2f}",
    }


def evaluate_classification_model(y_pred, y_test, model_name):
    """
    Evaluate a classification model and print detailed metrics.

    Args:
        y_pred: model prediction
        y_test: True labels for test data

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 scores (as percentages)
    """
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return ClassificationModelResult(
        accuracy=acc * 100,
        precision=prec * 100,
        recall=recall * 100,
        f1_score=f1 * 100,
        model_name=model_name,
    )

    return {
        "accuracy": f"{acc*100:.4f}",
        "precision": f"{prec*100:.2f}",
        "recall": f"{recall*100:.4f}",
        "f1 score": f"{f1*100:.4f}",
    }


def find_best_model(models_result: list, metric: str, higher_is_better: bool = True):
    if not models_result:
        return None

    if higher_is_better:
        return max(models_result, key=lambda m: getattr(m, metric))
    else:
        return min(models_result, key=lambda m: getattr(m, metric))


def calculate_cross_validation(model, X, y, random_state=42, cv=5, scoring="accuracy"):
    """
    This function calculates the Cross-Validation score for a given machine learning model.

    Args:
        model: The trained machine learning model instance (e.g., an SVC object).
        X: The feature data (input variables).
        y: The target labels (output variable).
        random_state: A random number.
        cv (int, optional): The number of folds for cross-validation. Defaults to 5.
        scoring (str, optional): The evaluation metric to use (e.g., 'accuracy', 'f1', 'roc_auc').
                                 Defaults to 'accuracy'.

    Returns:
        tuple: A tuple containing:
               - scores (numpy.ndarray): The calculated scores for each fold.
               - mean_score (float): The average score across all folds.
               - std_score (float): The standard deviation of the scores across all folds.
               - result_dict (dict): A dictionary containing detailed results per fold and overall.
    """

    # I check if the model has a predict_proba method, which is common for classifiers.
    if hasattr(model, "predict_proba") or hasattr(
        model, "classes_"
    ):  # Checking for classifier attributes
        cv_strategy = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
    else:
        cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Calculate Cross-Validation scores
    try:
        scores = cross_val_score(
            model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1
        )
    except ValueError as e:
        error_message = f"Error during cross-validation: {e}"
        print(error_message)
        return error_message, None, None, None

    # Calculate the mean and standard deviation of the scores
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Create a dictionary for more detailed results presentation
    result_dict = {f"Fold_{i+1}_{scoring}": score for i, score in enumerate(scores)}
    result_dict["Mean_Score"] = mean_score
    result_dict["Std_Score"] = std_score

    return scores, mean_score, std_score, result_dict


def plot_tf_history(history):
    """
    Plot training history from a TensorFlow model.

    Args:
        history: History object returned by model.fit()

    Returns:
        None: Displays two plots showing loss and accuracy over epochs
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    # Plot loss
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label="training_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def plot_torch_history(
    epochs: int, train_losses, val_losses, train_accs, val_accs: list[float]
):
    epochs_range = range(epochs)

    plt.figure(figsize=(14, 5))

    # ==== Loss Plot ====
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs_range, val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    # ==== Accuracy Plot ====
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label="Train Accuracy", linewidth=2)
    plt.plot(epochs_range, val_accs, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def make_window_horizon(
    df: pd.DataFrame, target: str, window: int, horizon: int, batch_size=None
):
    """
    This function will turn data into windows and horizons and return both.

    params:
        df: Input DataFrame
        target: Column name for the target variable
        window: Number of past timesteps to use as input
        horizon: Number of future timesteps to predict

    return:
        windows: Numpy array of input windows
        horizons: Numpy array of target horizons
        dataset: tf.data.Dataset instance
    """
    # Extract the target series
    series = df[target].values

    # Create windows (input features)
    windows = []
    horizons_list = []

    # Slide through the series to create windows and horizons
    for i in range(len(series) - window - horizon + 1):
        # Window: from i to i + window
        window_data = series[i : i + window]
        # Horizon: from i + window to i + window + horizon
        horizon_data = series[i + window : i + window + horizon]

        windows.append(window_data)
        horizons_list.append(horizon_data)

    # Convert to numpy arrays
    windows = np.array(windows)
    horizons = np.array(horizons_list)

    # Create the dataset using the same logic but with timeseries_dataset_from_array
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=series[:-horizon],  # Input data stops HORIZON steps before the end
        targets=horizons,  # The horizons we already created
        sequence_length=window,
        sequence_stride=1,
        shuffle=False,
        batch_size=batch_size,
    )

    return windows, horizons, dataset


def agument_images_yolo(images_path: Path, labels_path: Path, agument_count=1):
    """
    This function will agument you'r YOLO dataset

    params:
        images_path: Images folder path
        labels_path: Labels folder path
        agument_count: Agument count, default is 1

    """

    images_in_dir = os.listdir(images_path)
    labels_in_dir = os.listdir(labels_path)
    # in yolo datasets there is a file named `classes.txt` in label folder, we drop it from our list
    labels_in_dir.pop(labels_in_dir.index("classes.txt"))

    if not (len(images_in_dir) == len(labels_in_dir)):
        raise ValueError(
            "images count in image directory is not equal to labels count in label directory"
        )

    # you can change this transformer
    transformer = a.Compose(
        [
            a.Affine(  # this also known as shear
                p=0.2,
                shear=(-20, 20),
            ),
            a.HorizontalFlip(p=0.3),
            a.Rotate(p=0.2, limit=(-45, 45)),
            a.HueSaturationValue(p=0.15),
            a.ToGray(p=0.15),
            a.RandomBrightnessContrast(p=0.8, brightness_limit=(-0.3, 0.2)),
            a.GaussNoise(p=0.38),
            a.GaussianBlur(p=0.4, sigma_limit=(0.5, 1.5)),
        ],
        bbox_params=a.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    # we will agument dataset as much as `agument_count`
    count = 0
    while count < agument_count:
        for idx, (image_name, label_name) in enumerate(
            zip(images_in_dir, labels_in_dir)
        ):
            name = f"agu_{(idx + (len(images_in_dir) * count)):04}"
            image = cv2.imread(images_path / image_name)
            boxes = []
            labels = []

            # open label file and get it's `class, x_axis, y_axis, width, height`
            with open(labels_path / label_name) as f:
                for line in f:
                    if not line.strip():
                        continue
                    c, x, y, w, h = line.split()
                    boxes.append([float(x), float(y), float(w), float(h)])
                    labels.append(int(c))

            augmented = transformer(image=image, bboxes=boxes, class_labels=labels)

            new_boxes = augmented["bboxes"]
            new_labels = augmented["class_labels"]

            # save image and label
            cv2.imwrite((images_path / f"{name}.jpg"), augmented["image"])
            data = ""
            for cls, box in zip(new_labels, new_boxes):
                x, y, w, h = box
                data += f"{int(cls)} {x} {y} {w} {h}\n"

            with open(file=(labels_path / f"{name}.txt"), mode="w") as f:
                f.write(data)

        count += 1


__all__ = [
    "optional_import",
    "load_and_prep_image",
    "plot_images",
    "plot_roc_curve",
    "evaluate_regression_model",
    "evaluate_classification_model",
    "find_best_model",
    "calculate_cross_validation",
    "plot_tf_history",
    "plot_torch_history",
    "make_window_horizon",
    "agument_images_yolo",
    "run_grid_search",
]
