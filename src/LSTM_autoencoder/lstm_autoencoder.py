import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Final output layer
        self.output_layer = nn.Linear(input_size, input_size)
    
    def forward(self, x):
        # Encoder
        enc_output, (hidden, cell) = self.encoder(x)
        
        # Create decoder input sequence
        decoder_input = enc_output
        
        # Decoder
        dec_output, _ = self.decoder(decoder_input)
        
        # Final output
        output = self.output_layer(dec_output)
        
        return output