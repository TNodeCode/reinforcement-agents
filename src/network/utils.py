class NetworkUtils:
    """Helper methods for neural networks.
    """
    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Perform a soft update of the target network with the weights from the local network.

        Args:
            local_model (nn.Module): Local model
            target_model (nn.Module): Target model
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
