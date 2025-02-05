
class Accuracy:
    def __init__(self, topk=5):
        self.true = 0
        self.num_eval = 0

        self.topk = topk

    def reset(self):
        self.true = 0
        self.num_eval = 0

    def match(self, results, targets):
        num_batch = targets.size(0)

        # get top5 indices
        _, results_ = results.topk(self.topk, 1)
        targets_ = targets.unsqueeze(1).expand_as(results_)

        self.true = self.true + (results_ == targets_).sum().item()
        self.num_eval += num_batch

        print('....................................', self.true * 100. / self.num_eval)

    def get_result(self):
        return (self.true * 100. / self.num_eval)
