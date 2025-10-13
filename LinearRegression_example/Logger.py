
class Logger():
    
    def __init__(self,model_obj,trainer_obj):

        self.name = 'log.txt'
        self.model_obj = model_obj
        self.trainer_obj = trainer_obj
    
    def log(self,mode):

        with open(self.name,mode) as f:
            if mode == 'w':
                f.write('### Log file ###\n\n')
                f.write(f'ML model used: {self.model_obj.name}\n\n')
                f.write('Initial Parameters:\n')
                f.write(f'Weights: {str(self.model_obj.weights.tolist())}\n')
                f.write(f'Bias: {str(self.model_obj.bias.tolist())}\n')

            else:
                f.write('\n')
                f.write(f'Training algorithm: {self.trainer_obj.name}\n\n')
                f.write('Hyperparameters:\n')
                f.write(f'Eta: {str(self.trainer_obj.eta)}\n')
                f.write('Training results:\n\n')
                f.write(f'Weights: {str(self.model_obj.weights.tolist())}\n')
                f.write(f'Bias: {str(self.model_obj.bias.tolist())}\n')
                f.write(f'Loss function value after {str(self.trainer_obj.n_iter)} iterations is : {str(self.trainer_obj.losses[-1])}')
            