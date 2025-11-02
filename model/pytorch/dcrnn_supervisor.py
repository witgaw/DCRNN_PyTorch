import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from dcrnn_pytorch.lib import utils
from dcrnn_pytorch.lib.utils import DataLoader, StandardScaler
from dcrnn_pytorch.model.pytorch.dcrnn_model import DCRNNModel
from dcrnn_pytorch.model.pytorch.loss import masked_mae_loss
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNSupervisor:
    @staticmethod
    def _validate_data(data: Dict[str, Any]) -> None:
        """
        Validate the data dictionary structure.

        Args:
            data: Dictionary containing train/val/test loaders and scaler

        Raises:
            ValueError: If data structure is invalid
        """
        required_keys = ["train_loader", "val_loader", "test_loader", "scaler"]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            raise ValueError(f"Missing required keys in data: {missing_keys}")

        # Validate loaders are DataLoader instances
        for key in ["train_loader", "val_loader", "test_loader"]:
            if not isinstance(data[key], DataLoader):
                raise ValueError(
                    f"{key} must be an instance of dcrnn_pytorch.lib.utils.DataLoader, "
                    f"got {type(data[key])}"
                )

        # Validate scaler is StandardScaler instance
        if not isinstance(data["scaler"], StandardScaler):
            raise ValueError(
                f"scaler must be an instance of dcrnn_pytorch.lib.utils.StandardScaler, "
                f"got {type(data['scaler'])}"
            )

    def __init__(
        self, adj_mx, data_override: Optional[Dict[str, Any]] = None, **kwargs
    ):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get("data")
        self._model_kwargs = kwargs.get("model")
        self._train_kwargs = kwargs.get("train")

        self.max_grad_norm = self._train_kwargs.get("max_grad_norm", 1.0)

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter("runs/" + self._log_dir)

        log_level = self._kwargs.get("log_level", "INFO")
        self._logger = utils.get_logger(
            self._log_dir, __name__, "info.log", level=log_level
        )

        # data set
        if data_override is None:
            self._data = utils.load_dataset(**self._data_kwargs)
        else:
            self._validate_data(data_override)
            self._data = data_override
        self.standard_scaler = self._data["scaler"]

        self.num_nodes = int(self._model_kwargs.get("num_nodes", 1))
        self.input_dim = int(self._model_kwargs.get("input_dim", 1))
        self.seq_len = int(self._model_kwargs.get("seq_len"))  # for the encoder
        self.output_dim = int(self._model_kwargs.get("output_dim", 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get("use_curriculum_learning", False)
        )
        self.horizon = int(self._model_kwargs.get("horizon", 1))  # for the decoder

        # setup model
        dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        self.dcrnn_model = (
            dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model
        )
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get("epoch", 0)
        if self._epoch_num > 0:
            self.load_model()

        self._save_epoch = -1
        self._save_state = None

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs["train"].get("log_dir")
        if log_dir is None:
            batch_size = kwargs["data"].get("batch_size")
            learning_rate = kwargs["train"].get("base_lr")
            max_diffusion_step = kwargs["model"].get("max_diffusion_step")
            num_rnn_layers = kwargs["model"].get("num_rnn_layers")
            rnn_units = kwargs["model"].get("rnn_units")
            structure = "-".join(["%d" % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs["model"].get("horizon")
            filter_type = kwargs["model"].get("filter_type")
            filter_type_abbr = "L"
            if filter_type == "random_walk":
                filter_type_abbr = "R"
            elif filter_type == "dual_random_walk":
                filter_type_abbr = "DR"
            run_id = "dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/" % (
                filter_type_abbr,
                max_diffusion_step,
                horizon,
                structure,
                learning_rate,
                batch_size,
                time.strftime("%m%d%H%M%S"),
            )
            base_dir = kwargs.get("base_dir")
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def _get_config_for_saving(self, epoch):
        config = dict(self._kwargs)
        config["model_state_dict"] = self.dcrnn_model.state_dict()
        config["epoch"] = epoch
        # Save scaler state for inference
        config["scaler"] = {
            "mean": self.standard_scaler.mean,
            "std": self.standard_scaler.std,
        }
        return config

    def _load_from_checkpoint(self, checkpoint):
        self._setup_graph()
        self._save_state = checkpoint["model_state_dict"]
        self.dcrnn_model.load_state_dict(self._save_state)

        if "epoch" in checkpoint:
            self._epoch_num = checkpoint["epoch"]
            self._save_epoch = checkpoint["epoch"]

        # Restore scaler if available in checkpoint
        if "scaler" in checkpoint:
            self.standard_scaler = StandardScaler(
                mean=checkpoint["scaler"]["mean"], std=checkpoint["scaler"]["std"]
            )
            self._logger.info("Loaded model and scaler at {}".format(self._epoch_num))
        else:
            self._logger.warning(
                "Loaded model at {} but scaler not found in checkpoint. "
                "Using scaler from data loader.".format(self._epoch_num)
            )

    def save_model(self, epoch):
        if not os.path.exists("models/"):
            os.makedirs("models/")

        config = self._get_config_for_saving(epoch)

        torch.save(config, "models/epo%d.tar" % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return "models/epo%d.tar" % epoch

    def load_model(self):
        assert os.path.exists("models/epo%d.tar" % self._epoch_num), (
            "Weights at epoch %d not found" % self._epoch_num
        )
        checkpoint = torch.load(
            "models/epo%d.tar" % self._epoch_num, map_location="cpu"
        )

        self._load_from_checkpoint(checkpoint=checkpoint)

    def _get_model_path(self, save_dir, use_safetensors=True):
        if use_safetensors:
            return os.path.join(save_dir, "model.safetensors")
        else:
            return os.path.join(save_dir, "dcrnn_model.pth")

    def _get_config_path(self, save_dir):
        return os.path.join(save_dir, "config.json")

    def save_best_to_dir(self, save_dir, use_safetensors=True):
        """
        Save model to a custom directory in SafeTensors format (HuggingFace compatible).

        Args:
            save_dir: Directory path to save the model
            use_safetensors: If True, saves in SafeTensors format. If False, uses PyTorch format.

        Returns:
            str: Path to the saved model file
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get the full config
        config = (
            self._save_state
            if self._save_state
            else self._get_config_for_saving(self._epoch_num)
        )

        if use_safetensors:
            # Extract model state dict for SafeTensors
            model_state_dict = config["model_state_dict"]
            model_path = self._get_model_path(save_dir, use_safetensors=True)

            # Save model weights in SafeTensors format
            save_file(model_state_dict, model_path)

            # Save config metadata separately as JSON
            config_path = self._get_config_path(save_dir)
            metadata = {
                "epoch": config.get("epoch", self._epoch_num),
                "scaler": config.get("scaler"),
                "model_config": self._model_kwargs,
                "data_config": self._data_kwargs,
                "train_config": self._train_kwargs,
            }

            with open(config_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            self._logger.info(
                f"Saved model to {model_path} and config to {config_path}"
            )
            return model_path
        else:
            # Original PyTorch format
            model_path = self._get_model_path(save_dir, use_safetensors=False)
            torch.save(config, model_path)
            self._logger.info(f"Saved model to {model_path}")
            return model_path

    def load_from_dir(self, load_dir, use_safetensors=True):
        """
        Load model from a custom directory (supports both SafeTensors and PyTorch formats).

        Args:
            load_dir: Directory path containing the model file
            use_safetensors: If True, loads from SafeTensors format. If False, loads PyTorch format.
        """
        if use_safetensors:
            model_path = self._get_model_path(load_dir, use_safetensors=True)
            config_path = self._get_config_path(load_dir)

            if not os.path.exists(model_path) or not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"SafeTensors model file ({model_path}) or config file ({config_path}) not found"
                )

            # Load model weights
            model_state_dict = load_file(model_path)

            # Load config metadata
            with open(config_path, "r") as f:
                metadata = json.load(f)

            # Reconstruct checkpoint format
            checkpoint = {
                "model_state_dict": model_state_dict,
                "epoch": metadata.get("epoch"),
                "scaler": metadata.get("scaler"),
            }

            self._load_from_checkpoint(checkpoint=checkpoint)
            self._logger.info(f"Loaded model from {model_path}")
        else:
            # Original PyTorch format
            model_path = self._get_model_path(load_dir, use_safetensors=False)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            checkpoint = torch.load(model_path, map_location="cpu")
            self._load_from_checkpoint(checkpoint=checkpoint)
            self._logger.info(f"Loaded model from {model_path}")

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data["val_loader"].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset="val", batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data["{}_loader".format(dataset)].get_iterator()
            losses = []

            y_truths = []
            y_preds = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses)

            self._writer.add_scalar("{} loss".format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(
                y_truths, axis=1
            )  # concatenate on batch dimension

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            return mean_loss, {"prediction": y_preds_scaled, "truth": y_truths_scaled}

    def _train(
        self,
        base_lr,
        steps,
        patience=50,
        epochs=100,
        lr_decay_ratio=0.1,
        log_every=1,
        save_model=1,
        test_every_n_epochs=10,
        epsilon=1e-8,
        **kwargs,
    ):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float("inf")
        wait = 0
        optimizer = torch.optim.Adam(
            self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=steps, gamma=lr_decay_ratio
        )

        self._logger.info("Start training ...")

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data["train_loader"].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):
            self.dcrnn_model = self.dcrnn_model.train()

            train_iterator = self._data["train_loader"].get_iterator()
            losses = []

            start_time = time.time()

            for _, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x, y, batches_seen)

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(
                        self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon
                    )
                    # Recreate scheduler with the new optimizer
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, milestones=steps, gamma=lr_decay_ratio
                    )

                loss = self._compute_loss(y, output)

                self._logger.debug(loss.item())

                losses.append(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(
                    self.dcrnn_model.parameters(), self.max_grad_norm
                )

                optimizer.step()
            self._logger.info("epoch complete")
            self._logger.info("evaluating now!")

            val_loss, _ = self.evaluate(dataset="val", batches_seen=batches_seen)

            end_time = time.time()

            # Step the learning rate scheduler AFTER optimizer.step()
            lr_scheduler.step()

            self._writer.add_scalar("training loss", np.mean(losses), batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = (
                    "Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, "
                    "{:.1f}s".format(
                        epoch_num,
                        epochs,
                        batches_seen,
                        np.mean(losses),
                        val_loss,
                        lr_scheduler.get_last_lr()[0],
                        (end_time - start_time),
                    )
                )
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, _ = self.evaluate(dataset="test", batches_seen=batches_seen)
                message = (
                    "Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, "
                    "{:.1f}s".format(
                        epoch_num,
                        epochs,
                        batches_seen,
                        np.mean(losses),
                        test_loss,
                        lr_scheduler.get_last_lr()[0],
                        (end_time - start_time),
                    )
                )
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                self._save_epoch = epoch_num
                self._save_state = self._get_config_for_saving(epoch_num)
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        "Val loss decrease from {:.4f} to {:.4f}, saving to {}".format(
                            min_val_loss, val_loss, model_file_name
                        )
                    )
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning("Early stopping at epoch: %d" % epoch_num)
                    break

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., : self.output_dim].view(
            self.horizon, batch_size, self.num_nodes * self.output_dim
        )
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)

    def predict(self, x, batches_seen=0, apply_scaling=False):
        """
        Make predictions on input data.

        Args:
            x: Input data as numpy array of shape (batch_size, seq_len, num_nodes, input_dim).
               By default, expects SCALED data (same scale as training data).
            batches_seen: Number of batches seen (for curriculum learning, default: 0)
            apply_scaling: If True, applies standard scaling to input before prediction.
                          Set to True if input data is in ORIGINAL/UNSCALED form.
                          Default: False (assumes input is already scaled)

        Returns:
            numpy array of predictions in ORIGINAL/UNSCALED form, shape:
            (horizon, batch_size, num_nodes, output_dim)

        Example:
            # For already-scaled data (typical when loading from preprocessed dataset):
            predictions = supervisor.predict(x_scaled)

            # For raw/unscaled data:
            predictions = supervisor.predict(x_raw, apply_scaling=True)
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            # Validate input
            if not isinstance(x, np.ndarray):
                raise ValueError(f"Input must be a numpy array, got {type(x)}")

            expected_shape = (None, self.seq_len, self.num_nodes, self.input_dim)
            if x.ndim != 4 or x.shape[1:] != expected_shape[1:]:
                raise ValueError(
                    f"Input shape must be (batch_size, {self.seq_len}, {self.num_nodes}, {self.input_dim}), "
                    f"got {x.shape}"
                )

            # Apply scaling if requested
            if apply_scaling:
                x_scaled = x.copy()
                x_scaled[..., 0] = self.standard_scaler.transform(x[..., 0])
            else:
                x_scaled = x

            # Prepare data for model
            x_tensor = torch.from_numpy(x_scaled).float()
            x_tensor = x_tensor.permute(
                1, 0, 2, 3
            )  # (seq_len, batch_size, num_nodes, input_dim)
            batch_size = x_tensor.size(1)
            x_tensor = x_tensor.view(
                self.seq_len, batch_size, self.num_nodes * self.input_dim
            )
            x_tensor = x_tensor.to(device)

            # Make prediction (output is scaled)
            output = self.dcrnn_model(x_tensor, batches_seen=batches_seen)

            # Apply inverse scaling to get back to original scale
            output = self.standard_scaler.inverse_transform(output)

            # Convert to numpy and reshape
            output = output.cpu().numpy()
            output = output.reshape(
                self.horizon, batch_size, self.num_nodes, self.output_dim
            )

            return output
