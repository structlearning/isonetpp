SAVE = "save"
STOP = "stop"

class EarlyStopping:
    def __init__(self, patience=50, tolerance=0.0001):
        self.patience = patience
        self.tolerance = tolerance
        self.best_scores = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def diff(self, current_scores):
        return sum([current_score - best_score for current_score, best_score in zip(current_scores, self.best_scores)])
    
    def check(self, current_scores):
        verdict = None
        if self.best_scores is None:
            self.best_scores = current_scores
            verdict = SAVE
        elif self.diff(current_scores) >= self.tolerance:
            self.num_bad_epochs = 0
            self.best_scores = current_scores
            verdict = SAVE
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                verdict = STOP
        return verdict