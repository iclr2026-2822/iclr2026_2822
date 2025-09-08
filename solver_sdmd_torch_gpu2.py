import time
import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as la
from numpy import arange
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from functorch import vmap, jacrev

from torch.func import jacrev
import joblib
from sde_coefficients_estimator import SDECoefficientEstimator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

# device = 'cuda'
# device = 'cpu'

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class KoopmanNNTorch(nn.Module):
    def __init__(self, input_size, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
        super(KoopmanNNTorch, self).__init__()
        self.layer_sizes = layer_sizes
        self.n_psi_train = n_psi_train

        self.layers = nn.ModuleList()
        bias = False
        n_layers = len(layer_sizes)

        # First layer
        self.layers.append(nn.Linear(input_size, layer_sizes[0], bias=bias))
        # Hidden layers
        for ii in arange(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[ii], layer_sizes[ii+1], bias=True))
        # Activation and output layer
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(layer_sizes[-1], n_psi_train, bias=True))

    def forward(self, x):
        # 1) If input is a 1D vector, add batch dimension
        squeeze_back = False
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Convert to (1, D)
            squeeze_back = True

        # 2) Save original input
        in_x = x

        # 3) Normal forward pass
        for layer in self.layers:
            x = layer(x)

        # 4) Concatenate constant term, original input and network output
        const_out = torch.ones_like(in_x[:, :1])
        out = torch.cat([const_out, in_x, x], dim=1)

        # 5) If batch dimension was added at the beginning, remove it
        if squeeze_back:
            out = out.squeeze(0)  # Restore to original 1D

        return out
    


class KoopmanModelTorch(nn.Module):
    def __init__(self, dict_net, target_dim, k_dim):
        super(KoopmanModelTorch, self).__init__()
        self.dict_net = dict_net
        self.target_dim = target_dim
        self.k_dim = k_dim
        self.layer_K = nn.Linear(k_dim, k_dim, bias=False)
        self.layer_K.weight.requires_grad = False
    
    def forward(self, input_x, input_y):
        psi_x = self.dict_net.forward(input_x)
        psi_y = self.dict_net.forward(input_y)
        psi_next = self.layer_K(psi_x)
        outputs = psi_next - psi_y
        return outputs


class MLPModel(nn.Module):
    def __init__(self, num_features, num_outs, n_hid=128, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, n_hid),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid // 2),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid // 2),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 2, num_outs)
        )

        # 使用 Kaiming 初始化
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
        
class KoopmanSolverTorch(object):
    '''
    Build the Koopman solver

    This part represents a Koopman solver that can be used to build and solve Koopman operator models.

    Attributes:
        dic (class): The dictionary class used for Koopman operator approximation.
        dic_func (function): The dictionary functions used for Koopman operator approximation.
        target_dim (int): The dimension of the variable of the equation.
        reg (float, optional): The regularization parameter when computing K. Defaults to 0.0.
        psi_x (None): Placeholder for the feature matrix of the input data.
        psi_y (None): Placeholder for the feature matrix of the output data.
    '''

    def __init__(self, dic, target_dim, reg=0.0, checkpoint_file='example_koopman_net001.torch',
                 fnn_checkpoint_file='example_fnn001.torch', a_b_file=None,
                 generator_batch_size=4, fnn_batch_size=32, delta_t=0.1,
                 patience=7, min_delta=1e-4):
        """Initializer

        :param dic: dictionary
        :type dic: class
        :param target_dim: dimension of the variable of the equation
        :type target_dim: int
        :param reg: the regularization parameter when computing K, defaults to 0.0
        :type reg: float, optional
        """
        self.dic = dic  # dictionary class
        self.dic_func = dic.forward  # dictionary functions
        self.target_dim = target_dim
        self.reg = reg
        self.psi_x = None
        self.psi_y = None
        self.dPsi_X= None
        self.dPsi_Y= None
        self.checkpoint_file = checkpoint_file
        self.fnn_checkpoint_file=fnn_checkpoint_file
        self.generator_batch_size= generator_batch_size
        self.fnn_batch_size= fnn_batch_size
        self.delta_t= delta_t
        self.a_b_file= a_b_file
        self.patience = patience
        self.min_delta = min_delta

    def separate_data(self, data):
        data_x = data[0]
        data_y = data[1]
        return data_x, data_y

    def build(self, data_train):
        # Separate data
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)

        # Compute final information
        self.compute_final_info(reg_final=0.0)

    def compute_final_info(self, reg_final):
        # Compute K
        self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, reg=reg_final)
        self.K_np = self.K.detach().cpu().numpy()
        self.eig_decomp(self.K_np)
        #self.compute_mode()

    def eig_decomp(self, K):
        """ eigen-decomp of K """
        self.eigenvalues, self.eigenvectors = la.eig(K)
        idx = self.eigenvalues.real.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        self.eigenvectors_inv = la.inv(self.eigenvectors)

    def eigenfunctions(self, data_x):
        """ estimated eigenfunctions """
        data_x = torch.DoubleTensor(data_x).to(device)
        psi_x = self.dic_func(data_x)
        psi_x = psi_x.detach().cpu().numpy()
        val = np.matmul(psi_x, self.eigenvectors)
        return val

    # def compute_mode(self):
    #     self.basis_func_number = self.K.shape[0]

    #     # Form B matrix
    #     self.B = self.dic.generate_B(self.data_x_train)

    #     # Compute modes
    #     self.modes = np.matmul(self.eigenvectors_inv, self.B).T
    #     return self.modes

    # def calc_psi_next(self, data_x, K):
    #     psi_x = self.dic_func(data_x)
    #     psi_next = torch.matmul(psi_x, K)
    #     return psi_next

    def compute_K(self, dic, data_x, data_y, reg):
        data_x = torch.DoubleTensor(data_x).to(device)
        data_y = torch.DoubleTensor(data_y).to(device)
        psi_x = dic(data_x)
        psi_y = dic(data_y)
        
        # Compute Psi_X and Psi_Y
        self.Psi_X = dic(data_x)
        self.Psi_Y = dic(data_y)
        
        psi_xt = psi_x.T
        idmat = torch.eye(psi_x.shape[-1]).to(device)
        xtx_inv = torch.linalg.pinv(reg * idmat + torch.matmul(psi_xt, psi_x))
        xty = torch.matmul(psi_xt, psi_y)
        self.K_reg = torch.matmul(xtx_inv, xty)
        return self.K_reg

    def compute_K_with_generator(self): 
        """
        MODIFIED: Uses self.L_Psi (computed generator matrix L_N) directly.
        Arguments dic, data_x, data_y, reg are removed as they are no longer used.
        """
        
        if self.L_Psi is None:
            raise ValueError("Generator matrix L_Psi (L_N) must be computed before calling compute_K_with_generator.")

        # Direct use L_Psi to compute K = I + delta_t * L_N
        F = self.L_Psi.shape[0]
        device = self.L_Psi.device
        idmat = torch.eye(F, device=device, dtype=self.L_Psi.dtype)
        # SDMD Formula: K = I + delta_t * L_N
        self.K_gen = idmat + self.delta_t * self.L_Psi 

        return self.K_gen
    

    def get_Psi_X(self):
        return self.Psi_X

    def get_Psi_Y(self):
        return self.Psi_Y

    def build_model(self):
        self.koopman_model = KoopmanModelTorch(dict_net=self.dic, target_dim=self.target_dim, k_dim=self.K.shape[0]).to(device)
        
    # def fit_koopman_model(self, koopman_model, koopman_optimizer, checkpoint_file, xx_train, yy_train, xx_test, yy_test,
    #                   batch_size=32, lrate=1e-4, epochs=1000, initial_loss=1e15):
    #     load_best = False
    #     xx_train_tensor = torch.DoubleTensor((xx_train)).to(device)
    #     yy_train_tensor = torch.DoubleTensor((yy_train)).to(device)
    #     xx_test_tensor = torch.DoubleTensor((xx_test)).to(device)
    #     yy_test_tensor = torch.DoubleTensor((yy_test)).to(device)
    
    #     train_dataset = torch.utils.data.TensorDataset(xx_train_tensor, yy_train_tensor)
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    #     val_dataset = torch.utils.data.TensorDataset(xx_test_tensor, yy_test_tensor)
    #     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    #     n_epochs = epochs
    #     best_loss = initial_loss
    #     mlp_mdl = koopman_model
    #     #optimizer = torch.optim.Adam(mlp_mdl.parameters(), lr=lrate, weight_decay=1e-5)
    #     optimizer = koopman_optimizer
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lrate
    #     criterion = nn.MSELoss()
    
    #     mlp_mdl.train()
    #     val_loss_list = []
    
    #     for epoch in range(n_epochs):
    #         train_loss = 0.0
    #         for data, target in train_loader:
    #             optimizer.zero_grad()
    #             output = mlp_mdl(data, target)
    #             zeros_tensor = torch.zeros_like(output)
    #             loss = criterion(output, zeros_tensor)
    #             loss.backward()
    #             optimizer.step()
    #             train_loss += loss.item() * data.size(0)
    #         train_loss = train_loss / len(train_loader.dataset)
    
    #         val_loss = 0.0
    #         with torch.no_grad():
    #             for data, target in val_loader:
    #                 output_val = mlp_mdl(data, target)
    #                 zeros_tensor = torch.zeros_like(output_val)
    #                 loss = criterion(output_val, zeros_tensor)
    #                 val_loss += loss.item() * data.size(0)
    #         val_loss = val_loss / len(val_loader.dataset)
    #         val_loss_list.append(val_loss)
    
    #         print('Epoch: {} \tTraining Loss: {:.6f} val loss: {:.6f}'.format(
    #             epoch + 1, train_loss, val_loss))
    
    #         if val_loss < best_loss:
    #             print('saving, val loss enhanced:', val_loss, best_loss)
    #             #torch.save(mlp_mdl.state_dict(), checkpoint_file)
    #             torch.save({
    #             'model_state_dict': mlp_mdl.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             }, checkpoint_file)
    #             best_loss = val_loss
    #             load_best = True
    
    #     if load_best:
    #         #mlp_mdl.load_state_dict(torch.load(checkpoint_file))
    #         checkpoint = torch.load(checkpoint_file)
    #         mlp_mdl.load_state_dict(checkpoint['model_state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         mlp_mdl.layer_K.requires_grad = False
    #         koopman_model = mlp_mdl
    #         koopman_optimizer= optimizer
    
    #     return val_loss_list, best_loss

    def fit_koopman_model(self, koopman_model, koopman_optimizer, checkpoint_file, xx_train, yy_train,
                          xx_test, yy_test, batch_size=32, lrate=1e-4, epochs=1000, initial_loss=1e15):
        # Convert data to tensors and move to device
        from torch.cuda.amp import autocast, GradScaler

        # Create a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            koopman_optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Create an early stopping object
        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        # Create a scaler for mixed-precision training
        scaler = GradScaler()

        train_dataset = torch.utils.data.TensorDataset(
            torch.DoubleTensor(xx_train),
            torch.DoubleTensor(yy_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.DoubleTensor(xx_test),
            torch.DoubleTensor(yy_test)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        mlp_mdl = koopman_model
        criterion = nn.MSELoss()
        val_loss_list = []
        best_loss = initial_loss

        for epoch in range(epochs):
            train_loss = 0.0
            mlp_mdl.train()

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                koopman_optimizer.zero_grad()

                # Use mixed-precision training
                with autocast():
                    output = mlp_mdl(data, target)
                    zeros_tensor = torch.zeros_like(output)
                    loss = criterion(output, zeros_tensor)

                # Use the scaler for backpropagation
                scaler.scale(loss).backward()
                scaler.step(koopman_optimizer)
                scaler.update()

                train_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(train_loader.dataset)

            # Validation phase
            val_loss = 0.0
            mlp_mdl.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output_val = mlp_mdl(data, target)
                    zeros_tensor = torch.zeros_like(output_val)
                    loss = criterion(output_val, zeros_tensor)
                    val_loss += loss.item() * data.size(0)

            val_loss = val_loss / len(val_loader.dataset)
            val_loss_list.append(val_loss)

            # Update the learning rate scheduler
            scheduler.step(val_loss)

            print(f'Epoch: {epoch + 1} \tTraining Loss: {train_loss:.6f} val loss: {val_loss:.6f}')

            # Check if the best model needs to be saved
            if val_loss < best_loss:
                print(f'Saving model, val loss improved from {best_loss:.6f} to {val_loss:.6f}')
                torch.save({
                    'model_state_dict': mlp_mdl.state_dict(),
                    'optimizer_state_dict': koopman_optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_file)
                best_loss = val_loss

            # Check for early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Load the best model
        checkpoint = torch.load(checkpoint_file)
        mlp_mdl.load_state_dict(checkpoint['model_state_dict'])
        koopman_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        mlp_mdl.layer_K.requires_grad = False
        koopman_model = mlp_mdl
        koopman_optimizer = koopman_optimizer
        return val_loss_list, best_loss

    def train_psi(self, koopman_model, koopman_optimizer, epochs, lr, initial_loss=1e15):
        data_x_val, data_y_val = self.separate_data(self.data_valid)
        psi_losses, best_psi_loss = self.fit_koopman_model(self.koopman_model, koopman_optimizer, self.checkpoint_file, self.data_x_train,
                                                      self.data_y_train, data_x_val, data_y_val, self.batch_size,
                                                      lrate=lr, epochs=epochs, initial_loss=initial_loss)
        return psi_losses, best_psi_loss

                
    def process_batch(self, batch_inputs):
        """
        batch_inputs: Tensor of shape (B, D)
        Returns:
            J: Tensor of shape (B, F, D)
            H: Tensor of shape (B, F, D, D)
        """
        # 1) Compute the gradient function for a single sample x: x -> (F, D)
        jac_fn = jacrev(self.dic)
        # 2) Compute the Hessian function for a single sample x: x -> (F, D, D)
        hess_fn = jacrev(jac_fn)

        # 3) Vectorize over the batch dimension using vmap
        J = vmap(jac_fn)(batch_inputs)   # (B, F, D)
        H = vmap(hess_fn)(batch_inputs)  # (B, F, D, D)

        return J, H



    def compute_dPsi_X(self, data_x, b_Xt, a_Xt, delta_t):
        """
        Compute dPsi_X using batched derivatives, avoiding full Jacobian/Hessian storage.
        
        Args:
            data_x (Tensor): shape (M, D)
            b_Xt (Tensor): shape (M-1, D)
            a_Xt (Tensor): shape (M-1, D, D)
            delta_t (float): time step
            
        Returns:
            dPsi_X (Tensor): shape (M-1, F)
        """
        device = data_x.device
        num_samples = data_x.shape[0]
        num_features = self.dic(data_x[:1]).shape[1]  # F
        batch_size = 64
        num_batches = (num_samples + batch_size - 1) // batch_size

        dPsi_X = torch.zeros(num_samples - 1, num_features, device=device, dtype=data_x.dtype)
        batch_offset = 0

        for i, (batch_J, batch_H) in enumerate(self.get_derivatives(data_x, batch_size)):
            batch_size_actual = batch_J.shape[0]
            end_idx = min(batch_offset + batch_size_actual, num_samples - 1)

            batch_b = b_Xt[batch_offset:end_idx]
            batch_a = a_Xt[batch_offset:end_idx]

            term1 = torch.einsum('mfd,md->mf', batch_J[:end_idx - batch_offset], batch_b)
            term2 = 0.5 * torch.einsum('mfkl,mkl->mf', batch_H[:end_idx - batch_offset], batch_a)
            dPsi_X[batch_offset:end_idx] = term1 + term2

            batch_offset += batch_size_actual

        self.dPsi_X = dPsi_X
        return dPsi_X

    def get_derivatives(self, inputs, batch_size=64):
        """
        Yield batch-wise Jacobian and Hessian to avoid storing full tensors.

        Args:
            inputs (Tensor): shape (M, D)
            batch_size (int): size of each batch

        Yields:
            batch_J (Tensor): shape (batch_size, F, D)
            batch_H (Tensor): shape (batch_size, F, D, D)
        """
        with torch.no_grad():
            num_samples = inputs.shape[0]
            num_batches = (num_samples + batch_size - 1) // batch_size
            jac_fn = jacrev(self.dic)
            hess_fn = jacrev(jac_fn)

            for i in tqdm(range(num_batches), desc='Processing batches', unit='batch'):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_samples)
                batch_inputs = inputs[start:end]
                batch_J = vmap(jac_fn)(batch_inputs)  # (batch_size, F, D)
                batch_H = vmap(hess_fn)(batch_inputs)  # (batch_size, F, D, D)
                yield batch_J, batch_H
                del batch_J, batch_H
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()




    def compute_neural_a_b(self, data_x, delta_t):
        """
        Compute the drift and diffusion coefficients using the SDECoefficientEstimator.
        """
        num_samples, state_dim = data_x.shape
        X_t_1 = data_x[:-1, :].to(device)
        X_t = data_x[1:, :].to(device)
        
        # Initialize the SDE coefficient estimator
        sde_estimator = SDECoefficientEstimator(device=device)
        
        # Build the model with customizable parameters
        hidden_size = 128
        n_hidden_layers = 1
        dropout = 0.01
        
        sde_estimator.build_model(
            state_dim=state_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            n_hidden_layers=n_hidden_layers
        )
        
        # Train the model
        learning_rate = 5e-4
        epochs = 50 # 50
        batch_size = self.fnn_batch_size
        
        sde_estimator.fit_model(
            X_t_1=X_t_1,
            X_t=X_t,
            checkpoint_file=self.fnn_checkpoint_file,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs
        )
        
        # Estimate the coefficients
        b_Xt, a_Xt = sde_estimator.estimate_coefficients2(X_t_1, X_t, delta_t)
        
        # Handle a_Xt shape to ensure it's a 3D tensor even in 1D state space
        if state_dim == 1 and len(a_Xt.shape) == 2:
            # If 1D state space and a_Xt is a 2D tensor (M-1, 1)
            # Expand it to a 3D tensor (M-1, 1, 1)
            a_Xt_final = a_Xt.unsqueeze(-1)
            print(f"Expanded a_Xt shape from {a_Xt.shape} to {a_Xt_final.shape}")
        else:
            a_Xt_final = a_Xt
            print(f"Using original a_Xt shape: {a_Xt_final.shape}")
        
        return b_Xt, a_Xt_final

    def compute_generator_L(self, data_x, b_Xt, a_Xt, delta_t, lambda_reg=0.01):
        """
        Compute the generator matrix L via
          L = (PsiX^T PsiX + λI)^{-1} (PsiX^T dPsi_X)
        using a Cholesky solve and caching PsiX, instead of full pinv.
        """
        # 1) Move to GPU once
        data_x = data_x.to(device)

        # 2) Compute dPsi_X with your vectorized routine
        dPsi_X = self.compute_dPsi_X(data_x, b_Xt, a_Xt, delta_t)
        self.dPsi_X = dPsi_X

        # 3) Evaluate dictionary on all but last sample, store PsiX
        psi_x = self.dic(data_x[:-1])           # shape (M-1, F)
        
        # 4) Form Gram matrix G = PsiX^T PsiX  (F×F)
        G = psi_x.T @ psi_x
        
        # 5) Regularize
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        G_reg = G + lambda_reg * I
        
        # 6) Compute RHS A = PsiX^T @ dPsi_X   (F×F)
        A = psi_x.T @ dPsi_X

        # 7) Solve G_reg · L_Psi = A via Cholesky (SPD solve)
        #    This is much faster and more stable than pinv:
        #    G_reg = L L^T
        L = torch.linalg.cholesky(G_reg)        # lower-triangular L
        L_Psi = torch.cholesky_solve(A, L)      # solves L L^T X = A

        # 8) Cache and return
        self.L_Psi = L_Psi
        return L_Psi

    def build_with_generator(self, data_train, data_valid, epochs, batch_size, lr, log_interval, lr_decay_factor):
        """Train Koopman model and calculate the final information,
        such as eigenfunctions, eigenvalues and K.
        For each outer training epoch, the koopman dictionary is trained
        by several times (inner training epochs), and then compute matrix K.
        Iterate the outer training.

        :param data_train: training data
        :type data_train: [data at the current time, data at the next time]
        :param data_valid: validation data
        :type data_valid: [data at the current time, data at the next time]
        :param epochs: the number of the outer epochs
        :type epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param lr: learning rate
        :type lr: float
        :param log_interval: the patience of learning decay
        :type log_interval: int
        :param lr_decay_factor: the ratio of learning decay
        :type lr_decay_factor: float
        """
        # Separate training data
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)

        self.data_valid = data_valid

        self.batch_size = batch_size
        data_x_train_tensor= torch.DoubleTensor(self.data_x_train)
        #here we load compute drift and diffusion coefficents using feed-forward neural network 
        self.b_Xt, self. a_Xt = self.compute_neural_a_b(data_x_train_tensor, delta_t= self.delta_t)
        self. L_Psi = self.compute_generator_L(data_x_train_tensor, self.b_Xt, self.a_Xt, self.delta_t)
        self.K = self.compute_K_with_generator()
        # here we save drift and diffusion coefficents to  the joblib file, if filename  is specified.
        if (self.a_b_file is not None):
            a_Xt_np= self.a_Xt.detach().cpu().numpy()
            b_Xt_np= self.b_Xt.detach().cpu().numpy()
            print ('saving FNN a and b to: ', self.a_b_file )
            joblib.dump ((a_Xt_np,b_Xt_np), self.a_b_file)

        # Modify the initialization of the learning rate optimizer and scheduler.
        if not hasattr(self, 'koopman_model') or self.koopman_model is None:
            self.build_model()
            dict_params = [p for n, p in self.koopman_model.named_parameters()
                           if "layer_K.weight" not in n]
            self.koopman_optimizer = torch.optim.AdamW(  # Use AdamW optimizer
                dict_params, lr=lr, weight_decay=1e-5
            )
        else:
            # Subsequent times: check dimensions, then update layer_K
            assert self.K.shape[0] == self.koopman_model.k_dim, "K matrix dimensions mismatch"

        # # In either case, always assign the latest K to layer_K and freeze it
        # with torch.no_grad():
        #     self.koopman_model.layer_K.weight.copy_(self.K)
        # self.koopman_model.layer_K.weight.requires_grad = False

        losses = []
        curr_lr = lr
        curr_last_loss = 1e15
        # self.koopman_optimizer = torch.optim.Adam(self.koopman_model.parameters(), lr=lr, weight_decay=1e-5)
        for ii in arange(epochs):
            #starting outer epoch. In each outer epoch we compute generator L
            #Koopman operator K is computed from L each outer epoch,
            # and the matrix K is set as weighths of layer K of our Koopman NN. 
            #then we do several steps of training our NN that is the dictionary
            start_time = time.time()
            print(f"Outer Epoch {ii+1}/{epochs}")
            
            # One step for computing L and K
            self. L_Psi = self.compute_generator_L(data_x_train_tensor, self.b_Xt, self. a_Xt, self.delta_t)
            self.K = self.compute_K_with_generator()
            
            with torch.no_grad():
                # self.koopman_model.layer_K.weight.data = self.K
                self.koopman_model.layer_K.weight.data.copy_(self.K) # self.K.T
            self.koopman_model.layer_K.weight.requires_grad = False

            # steps (inner epochs) for training PsiNN, the number of inner epochs is given by epochs parameter below, here epochs= 4
            curr_losses, curr_best_loss = self.train_psi(
                self.koopman_model,
                self.koopman_optimizer,
                epochs=2,
                lr=curr_lr,
                initial_loss=curr_last_loss
            )
            
            if curr_last_loss > curr_best_loss:
                curr_last_loss = curr_best_loss

            if ii % log_interval == 0:
                losses.append(curr_losses[-1])

                # Adjust learning rate:
                if len(losses) > 2:
                    if losses[-1] > losses[-2]:
                        print("Error increased. Decay learning rate")
                        curr_lr = lr_decay_factor * curr_lr

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {ii+1} time: {epoch_time:.2f} seconds")

        # Compute final information
        #self.koopman_model.load_state_dict(torch.load(self.checkpoint_file))
        checkpoint = torch.load(self.checkpoint_file)
        self.koopman_model.load_state_dict(checkpoint['model_state_dict'])
        self.koopman_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.compute_final_info(reg_final=0.01)