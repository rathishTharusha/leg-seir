# Terms used in this project

| Term | Meaning here |
|---|---|
| Baseline | A comparison method, often simpler than the proposed model. |
| Persistence | Predict the latest known count again at every future horizon. |
| MLP | Multilayer perceptron: a neural network made of connected layers. |
| GNN | Graph neural network: a network that combines information using graph links. |
| Graph | Districts are nodes; weighted directed links describe which districts supply information to others. |
| Lag | A past time position. We have three input report positions. |
| Horizon | A future report position to predict. |
| Forecast origin | The point where a forecast starts. Only reports before this point are inputs. |
| Sample / window | Three input reports and the three reports we want to predict. |
| Batch | A group of samples processed together during training. |
| Tensor | An array, often with several dimensions, used by PyTorch. |
| Hidden vector | A list of learned numbers representing the input. |
| Parameter / weight | A model value adjusted during training. |
| Embedding | A small learned vector representing an item such as a district. |
| ReLU | An activation that keeps positive inputs and changes negative inputs to zero. |
| Softmax | Converts scores to positive weights that sum to one. |
| Sigmoid / gate | Converts a number to a value between zero and one; a gate controls how much of a signal is used. |
| Gradient | How the loss changes when a model parameter changes. |
| Learning rate / LR | The step size used to update parameters. |
| Epoch | One pass through the training samples. |
| Random seed | A starting value for random-number generation. Different seeds can give different fitted models. |
| Checkpoint | Saved model weights and settings from a training run. |
| Early stopping / patience | Stop after validation error has failed to improve for a chosen number of epochs. |
| Training split | Data used to update model weights. |
| Validation split | Data used to choose settings or the best checkpoint. |
| Test split | Data used to measure final prediction error. Prior research use of our 2026 sample limits its independence. |
| Fold | One training/validation split. Development folds validate on 2023 and 2025. |
| MSE | Mean squared error: average of squared prediction errors. |
| RMSE | Square root of MSE. Large errors count more heavily. Lower is better. |
| MAE | Mean absolute error: average size of errors, ignoring their signs. Lower is better. |
| sMAPE | A percentage-style error measure. It needs care when counts are near zero. |
| Seed SD | Standard deviation of scores across training seeds; it is not a confidence interval. |
| Ensemble | A forecast made by combining predictions from separately trained models. |
| Standardization | Subtract a mean and divide by a standard deviation. We calculate them from training data only. |
| SEIR | Susceptible, exposed, infectious and recovered groups in a disease model. Our states are numerical proxies. |
| Proxy | A model value used in place of a quantity that was not directly measured. |
| Imported pressure | A graph-based input from other districts used in the disease calculation; not measured travel. |
| Ablation / removal control | A separately trained model with a component removed or replaced. It tests the changed model, not every mechanism separately. |
| Sensitivity check | Repeat a comparison after changing a chosen setting. |
| Retrospective | A study using past data. Here, earlier inspection of 2026 informed some research choices. |
| Prospective | An evaluation planned before the future outcomes are observed. |
| Hash / SHA256 | A file fingerprint used to check whether bytes have changed. |
| Pinned revision | The exact saved version of a repository or dataset used by a download. |
| CPU checkpoint parity | Reload saved weights on CPU and check that predictions agree within a stated numerical tolerance. |
| Reproduction | Run a clearly specified experiment again and compare its outputs, while recording differences. |

Small floating-point differences can occur across computers. Matching saved predictions is different from training a new model and getting the same score.
