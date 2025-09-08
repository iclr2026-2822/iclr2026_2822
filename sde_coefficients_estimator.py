import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class MLPModel(nn.Module):
    """
    Multi-layer perceptron model for estimating SDE coefficients.
    
    Args:
        num_features (int): Number of input features (state dimension)
        num_outs (int): Number of output features
        n_hid (int): Number of neurons in the hidden layer
        dropout (float): Dropout rate for regularization
    """
    def __init__(self, num_features, num_outs, n_hid=128, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid, num_outs),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)


class SDECoefficientEstimator:
    """
    Estimates drift and diffusion coefficients for stochastic differential equations.
    
    This class uses a neural network to approximate the drift coefficient and
    calculates the diffusion coefficient based on the residuals.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.fnn_model = None
        self.optimizer = None
        torch.set_default_dtype(torch.float64)
        
    def build_model(self, state_dim, hidden_size=128, dropout=0.1, n_hidden_layers=1):
        """
        Build the neural network model for SDE coefficient estimation.
        
        Args:
            state_dim (int): Dimension of the state space
            hidden_size (int): Number of neurons in each hidden layer
            dropout (float): Dropout rate for regularization
            n_hidden_layers (int): Number of hidden layers
        """
        if n_hidden_layers == 1:
            self.fnn_model = MLPModel(
                num_features=state_dim, 
                num_outs=state_dim, 
                n_hid=hidden_size, 
                dropout=dropout
            ).to(self.device)
        else:
            # Create a custom model with multiple hidden layers
            layers = [nn.Linear(state_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            
            # Add additional hidden layers
            for _ in range(n_hidden_layers - 1):
                layers.extend([
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                
            # Add output layer
            layers.append(nn.Linear(hidden_size, state_dim))
            
            # Create the model
            self.fnn_model = nn.Sequential(*layers).to(self.device)
            
            # Initialize weights
            for m in self.fnn_model:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
        
        return self.fnn_model
    
    def fit_model(self, X_t_1, X_t, checkpoint_file, batch_size=32, learning_rate=1e-4, 
                  epochs=1000, weight_decay=1e-5, test_size=0.2, initial_loss=10000):
        """
        Train the neural network to predict the next state given the current state.
        
        Args:
            X_t_1 (torch.Tensor or numpy.ndarray): Current state tensor
            X_t (torch.Tensor or numpy.ndarray): Next state tensor
            checkpoint_file (str): Path to save the best model checkpoint
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimization
            epochs (int): Number of training epochs
            weight_decay (float): Weight decay for regularization
            test_size (float): Proportion of data to use for validation
            initial_loss (float): Initial loss value for comparing improvement
            
        Returns:
            tuple: (validation_loss_history, best_validation_loss)
        """
        # Convert inputs to tensors if they aren't already
        if not torch.is_tensor(X_t_1):
            X_t_1 = torch.DoubleTensor(X_t_1).to(self.device)
            X_t = torch.DoubleTensor(X_t).to(self.device)
        else:
            X_t_1 = X_t_1.to(self.device)
            X_t = X_t.to(self.device)
        
        # Split data into training and validation sets
        xx_train, xx_test, yy_train, yy_test = train_test_split(X_t_1, X_t, test_size=test_size)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(xx_train, yy_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(xx_test, yy_test)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.fnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Training loop
        best_loss = initial_loss
        val_loss_list = []
        load_best = False
        
        for epoch in range(epochs):
            # Training phase
            self.fnn_model.train()
            train_loss = 0.0
            
            for data, target in train_loader:
                self.optimizer.zero_grad()
                output = self.fnn_model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * data.size(0)
                
            train_loss = train_loss / len(train_loader.dataset)
            
            # Validation phase
            self.fnn_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output_val = self.fnn_model(data)
                    loss = criterion(output_val, target)
                    val_loss += loss.item() * data.size(0)
                    
            val_loss = val_loss / len(val_loader.dataset)
            val_loss_list.append(val_loss)
            
            print(f'Epoch: {epoch + 1} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {val_loss:.6f}')
            
            # Save the best model
            if val_loss < best_loss:
                print(f'Saving model, validation loss improved: {val_loss:.6f} < {best_loss:.6f}')
                torch.save({
                    'model_state_dict': self.fnn_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_file)
                best_loss = val_loss
                load_best = True
        
        # Load the best model
        if load_best and checkpoint_file:
            checkpoint = torch.load(checkpoint_file)
            self.fnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return val_loss_list, best_loss
    
    def estimate_coefficients(self, X_t_1, X_t, delta_t):
        """
        Estimate drift and diffusion coefficients from the data.
        
        Args:
            X_t_1 (torch.Tensor): Current state tensor
            X_t (torch.Tensor): Next state tensor
            delta_t (float): Time step
            
        Returns:
            tuple: (b_Xt, a_Xt_final) - drift and diffusion coefficients
        """
        # Ensure model is in evaluation mode
        self.fnn_model.eval()
        
        # Move data to device if needed
        X_t_1 = X_t_1.to(self.device)
        X_t = X_t.to(self.device)
        
        # Predict next state (drift term)
        with torch.no_grad():
            b_Xt = self.fnn_model(X_t_1)
        
        # Calculate residuals
        residuals = X_t - b_Xt
        
        # Compute variance of residuals
        variance = torch.square(residuals)
        
        # Calculate diffusion coefficient
        a_Xt = torch.sqrt(variance / delta_t)
        
        # Convert to diagonal matrices for multivariate case
        state_dim = a_Xt.shape[1]
        if state_dim > 1:
            a_xt_diags = []
            for jj in range(a_Xt.shape[0]):
                a_xt_diags.append(torch.diag(a_Xt[jj, :].squeeze()))
            a_Xt_final = torch.stack(a_xt_diags)
        else:
            a_Xt_final = a_Xt
            
        return b_Xt, a_Xt_final
    
    def estimate_coefficients2(self, X_t_1, X_t, delta_t):
        """
        Estimate the drift b(x) and diffusion matrix σσᵀ for the SDE.

        Args:
            X_t_1 (Tensor): current state of shape (N, D)
            X_t (Tensor): next state of shape (N, D)
            delta_t (float): time step Δt

        Returns:
            b_Xt (Tensor): drift term of shape (N, D)
            diffusion_matrix (Tensor): diffusion matrix σσᵀ of shape (N, D, D)
        """
        self.fnn_model.eval()
        X_t_1 = X_t_1.to(self.device)
        X_t = X_t.to(self.device)

        # Predict drift b(x)
        with torch.no_grad():
            b_Xt = self.fnn_model(X_t_1)

        # Compute residuals and variance
        residuals = X_t - b_Xt
        variance = residuals.pow(2)

        # Compute σ = sqrt(variance / delta_t)
        sigma = torch.sqrt(variance / delta_t)

        # Build diagonal matrices of σ for each sample
        N, D = sigma.shape
        sigma_mats = torch.stack([torch.diag(sigma[i]) for i in range(N)])  # (N, D, D)

        # Compute diffusion matrix σσᵀ
        diffusion_matrix = sigma_mats @ sigma_mats.transpose(-1, -2)

        return b_Xt, diffusion_matrix

    
    def load_model(self, checkpoint_file):
        """
        Load a trained model from a checkpoint file.
        
        Args:
            checkpoint_file (str): Path to the model checkpoint
        """
        if self.fnn_model is None:
            raise ValueError("Model must be built before loading weights")
            
        checkpoint = torch.load(checkpoint_file)
        self.fnn_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create optimizer if it doesn't exist
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.fnn_model.parameters())
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fnn_model.eval()  # Set to evaluation mode